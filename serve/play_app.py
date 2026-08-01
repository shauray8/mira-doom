import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATASETS, latest_checkpoint_path, run_dir  # noqa: E402

# Which checkpoint to play. The single-player base (wm, 75k) is the most thoroughly trained; the
# long-horizon finetune (wm_long) can be swapped in via PLAY_STAGE to compare drift behaviour.
PLAY_STAGE = "wm"
N_DIFFUSION_STEPS = 2  # fewer = faster, lower quality; 2 is the interactive floor
NOISE_LEVEL = 0.45  # the drift-suppressing value the sweep found
# Classifier-free guidance strength on the ACTION (1.0 = off, plain conditional). >1 costs a second
# DiT forward per diffusion step but makes the world follow the controller instead of its own prior.
GUIDANCE = 1.0
# The 2x lever: the single-frame forward is launch-overhead bound and the graph collapses that.
# Requires SINGLE_PLAYER (fixed buffers) and NOT compiling decode (captured inside the graph).
USE_CUDA_GRAPH = True
COMPILE_DECODE = not USE_CUDA_GRAPH  # fallback path only
SINGLE_PLAYER = True

DOOM_KEYS = [
    "forward", "backward", "strafe_right", "strafe_left",
    "weapon1", "weapon2", "weapon3", "weapon4", "weapon5", "weapon6", "weapon7",
    "attack", "speed",
]  # fmt: skip
JPEG_QUALITY = 72

WEAPON_HOLD_STEPS = 2
ATTACK_MIN_STEPS = 3
INITIAL_WEAPON_STEPS = 6  # longer at session start: it must register against a context seeded
                          # holding some other weapon
WEAPON_SLICE = slice(4, 11)  # weapon1..weapon7 in DOOM_KEYS

_GPU_JPEG = None  # None = untried, True = nvJPEG works, False = fall back to PIL


def encode_jpeg(frame_hwc):
    global _GPU_JPEG
    if _GPU_JPEG is not False:
        try:
            from torchvision.io import encode_jpeg as _tv_encode

            chw = frame_hwc.permute(2, 0, 1).contiguous()
            jpeg = _tv_encode(chw, quality=JPEG_QUALITY)  # CUDA in -> nvJPEG
            _GPU_JPEG = True
            return jpeg.cpu().numpy().tobytes()
        except Exception:  # noqa: BLE001 -- no nvJPEG in this torchvision build; use PIL from here on
            _GPU_JPEG = False
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(frame_hwc.cpu().numpy()).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


class InteractivePlayer:
    def __init__(
        self,
        model,
        seed_batch,
        *,
        n_diffusion_steps: int,
        noise_level: float,
        device: str,
        guidance: float = 1.0,
    ):
        import torch

        from mira.world_model.config import WorldModelInferenceConfig

        self.torch = torch
        self.model = model
        self.device = device
        self.guidance = guidance
        self._null_action = self._build_null_action() if guidance != 1.0 else None
        self.cfg = WorldModelInferenceConfig(n_diffusion_steps=n_diffusion_steps, noise_level=noise_level)
        self.atd = model.action_temporal_downsampling
        self.n_ctx = model.n_context_latents
        self.window_size = self.n_ctx + 1
        self._seed_batch = seed_batch
        self._seed(seed_batch)

    def _build_null_action(self):
        torch = self.torch
        enc = self.model.action_encoder
        if enc.mouse_dropout_token is None or enc.keyboard_dropout_token is None:
            return None
        with torch.no_grad():
            tok = torch.cat([enc.mouse_dropout_token, enc.keyboard_dropout_token], dim=-1)
            return enc.joint_mlp(tok.to(next(enc.joint_mlp.parameters()).dtype)).clone()

    def _guided(self, m, z_last, tau, clean_past):
        pred_c = m(z_last, self._g_action, tau, kv_caches=self.kv, clean_past=clean_past)
        if self.guidance == 1.0 or self._null_action is None:
            return pred_c
        pred_u = m(z_last, self._g_null, tau, kv_caches=self.kv, clean_past=clean_past)
        return pred_u + self.guidance * (pred_c - pred_u)

    def reset(self) -> None:
        self._seed(self._seed_batch)

    def reseed_in_place(self, seed_batch=None) -> None:
        torch = self.torch
        fixed = (self.z_t, self.kv, self.a_keys, self.a_mouse)

        self._seed(self._seed_batch if seed_batch is None else seed_batch)
        self.step([0] * self.n_keys, 0.0)  # eager: builds the KV-cache from the fresh context

        z_buf, kv_buf, keys_buf, mouse_buf = fixed
        if kv_buf is not None:  # graph captured: copy the fresh state into its buffers
            with torch.no_grad():
                z_buf.copy_(self.z_t)
                keys_buf.copy_(self.a_keys)
                mouse_buf.copy_(self.a_mouse)
                for i, slot in enumerate(kv_buf):
                    if slot is None:
                        continue
                    slot[0].copy_(self.kv[i][0])
                    slot[1].copy_(self.kv[i][1])
            self.z_t, self.kv, self.a_keys, self.a_mouse = z_buf, kv_buf, keys_buf, mouse_buf

    def _denoise_body(self):
        """One steady-state denoise+decode, entirely in-place on fixed buffers. Reimplements
        LatentWorldModel.denoise_streaming's steady-state; see the notes there."""
        torch = self.torch
        m = self.model.world_model
        z_last = self.z_t[:, -1:]
        z_last.copy_(self._g_noise)  # start the new frame from noise (static input)
        clean_past = self.z_t[:, -2:-1]

        for tau_t, dt in zip(self._taus, self._dts):
            z_last.add_(dt * self._guided(m, z_last, tau_t, clean_past))

        nl = self.cfg.noise_level
        current_z = (1.0 - nl) * z_last + nl * self._g_renoise
        _, new_kv = m(
            current_z, self._g_action, self._tau_renoise,
            kv_caches=self.kv, return_kv=True, clean_past=clean_past,
        )
        for i in range(len(self.kv)):
            if self.kv[i] is None:  # non-temporal-attention layers don't cache (time_attention_every)
                continue
            k_buf, v_buf = self.kv[i]
            nk, nv = new_kv[i]
            k_buf[:, :-1].copy_(k_buf[:, 1:].clone())  # drop oldest (n_register_tokens=0)
            k_buf[:, -1:].copy_(nk)  # append current
            v_buf[:, :-1].copy_(v_buf[:, 1:].clone())
            v_buf[:, -1:].copy_(nv)

        frame = self.model.decode_to_video(z_last)
        self._g_frame.copy_((frame[0].clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1))
        self.z_t[:, :-1].copy_(self.z_t[:, 1:].clone())

    def setup_graph(self):
        """Warm up to steady state, freeze the KV-cache into fixed buffers, then capture the step."""
        torch = self.torch
        from mira.world_model.schedule import build_inference_schedule

        for _ in range(6):  # populate the KV-cache and stabilise before capture
            self.step([0] * self.n_keys, 0.0)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            self.kv = [
                (kv[0].contiguous().clone(), kv[1].contiguous().clone()) if kv is not None else None
                for kv in self.kv
            ]
            self.z_t = self.z_t.contiguous().clone()

            z_last = self.z_t[:, -1:]
            self._g_noise = torch.randn_like(z_last)
            self._g_renoise = torch.randn_like(z_last)
            self._g_action = self._encode_actions()[:, -1:].contiguous().clone()  # (1,1,d)
            self._g_null = self.torch.zeros_like(self._g_action)
            if self._null_action is not None:
                self._g_null.copy_(self._null_action.expand_as(self._g_null))
            probe = self.model.decode_to_video(z_last)  # (1, T_video, C, H, W)
            self.n_video_frames = probe.shape[1]
            H, W = probe.shape[-2:]
            self._g_frame = torch.zeros(
                (self.n_video_frames, H, W, 3), dtype=torch.uint8, device=self.device
            )

            ts = build_inference_schedule(self.cfg.n_diffusion_steps, self.device, self.cfg.schedule_type)
            dts = ts[1:] - ts[:-1]
            ones = torch.ones((1, 1, 1, 1, 1), device=self.device, dtype=self.z_t.dtype)
            self._taus = [t * ones for t in ts[:-1]]
            self._dts = [float(d) for d in dts]
            self._tau_renoise = (1.0 - self.cfg.noise_level) * ones

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._denoise_body()
            torch.cuda.current_stream().wait_stream(s)
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                self._denoise_body()

    def step_fast(self, key_vec, turn: float):
        torch = self.torch
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            self._g_action.copy_(self._encode_action_last(key_vec, turn))
            self._g_noise.normal_()
            self._g_renoise.normal_()
            self._graph.replay()
            return [encode_jpeg(self._g_frame[i]) for i in range(self._g_frame.shape[0])]

    def _encode_action_last(self, key_vec, turn: float):
        """Encode the live action and return just the last latent-frame's conditioning (1,1,d)."""
        torch = self.torch
        kp = torch.tensor(key_vec, dtype=torch.int32, device=self.device).view(1, 1, -1).repeat(1, self.atd, 1)
        mv = torch.zeros((1, self.atd, 2), dtype=torch.float32, device=self.device)
        mv[:, :, 0] = turn
        self.a_keys = torch.cat([self.a_keys[:, self.atd :], kp], dim=1).contiguous()
        self.a_mouse = torch.cat([self.a_mouse[:, self.atd :], mv], dim=1).contiguous()
        return self._encode_actions()[:, -1:]

    def _seed(self, seed_batch) -> None:
        torch = self.torch
        self.model.codec.preprocess_batch(seed_batch)
        seed_batch = seed_batch.to(self.device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z = self.model.encode_video(seed_batch).clone()  # (1, T, h, w, c)

        # z_t: [ context_0 .. context_{W-2}, noise ]  (window_size frames)
        ctx = z[:, : self.window_size - 1]
        noise = torch.randn((1, 1, *z.shape[2:]), device=self.device, dtype=z.dtype)
        self.z_t = torch.cat([ctx, noise], dim=1).contiguous()
        # actions: the matching window_size*atd steps as float/int tensors we slide directly.
        a = seed_batch.actions
        self.action_config = a.config
        self.n_keys = a.key_presses.shape[-1]
        self.a_keys = a.key_presses[:, : self.window_size * self.atd].to(self.device).contiguous()
        self.a_mouse = a.mouse_movements[:, : self.window_size * self.atd].to(self.device).contiguous()
        # Keep on-device: the encoder reads game_mouse_sensitivity too; a fresh ActionTensors would
        # default it to a CPU tensor and hit a device mismatch inside the encoder's linear.
        self.a_sens = a.game_mouse_sensitivity.to(self.device)
        self.kv = None  # first step builds the cache from the context

    def _encode_actions(self):
        """Action conditioning for the current fixed window (start is always 0)."""
        from mira.world_model.actions_config import ActionTensors

        off = self.atd - 1
        a = ActionTensors(config=self.action_config, batch_size=1)
        a.key_presses = self.a_keys[:, off : (self.window_size - 1) * self.atd + off]
        a.mouse_movements = self.a_mouse[:, off : (self.window_size - 1) * self.atd + off]
        a.game_mouse_sensitivity = self.a_sens
        return self.model.action_encoder(a).clone()

    def step(self, key_vec, turn: float):
        """Advance one latent frame under the given action; return a (T_video, H, W, 3) uint8 numpy
        array of the video frames this latent step decodes to (chronological)."""
        torch = self.torch

        # Slide the action buffer: drop the oldest atd steps, append atd copies of the live action.
        kp = torch.tensor(key_vec, dtype=torch.int32, device=self.device).view(1, 1, -1).repeat(1, self.atd, 1)
        mv = torch.zeros((1, self.atd, 2), dtype=torch.float32, device=self.device)
        mv[:, :, 0] = turn
        self.a_keys = torch.cat([self.a_keys[:, self.atd :], kp], dim=1).contiguous()
        self.a_mouse = torch.cat([self.a_mouse[:, self.atd :], mv], dim=1).contiguous()

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            current_a = self._encode_actions()
            # NOTE: the eager path goes through denoise_streaming, which has no guidance hook, so
            # guidance applies on the graphed path only. The graphed path is the one that serves.
            self.z_t, self.kv = self.model.denoise_streaming(
                self.z_t, current_a,
                n_diffusion_steps=self.cfg.n_diffusion_steps,
                noise_level=self.cfg.noise_level,
                streaming_kv_caches=self.kv,
                schedule_type=self.cfg.schedule_type,
            )
            new_latent = self.z_t[:, -1:]  # the frame just generated
            frame = self.model.decode_to_video(new_latent)  # (1, 1, C, H, W) in [0,1]

            # Slide z_t: drop oldest, append a fresh noise frame for the next step. The just-generated
            # clean frame becomes the new clean_past; shape stays window_size (static).
            noise = torch.randn((1, 1, *self.z_t.shape[2:]), device=self.device, dtype=self.z_t.dtype)
            self.z_t = torch.cat([self.z_t[:, 1:], noise], dim=1).contiguous()

        return (frame[0].clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


class Session:
    def __init__(self, sid, player):
        import collections
        import threading

        self.sid = sid
        self.player = player
        self.keys = [0] * len(DOOM_KEYS)
        self.turn = 0.0
        self.reset_req = False
        import random

        # Random starting weapon on every new session (i.e. every page refresh).
        self.weapon_idx = random.randrange(WEAPON_SLICE.start, WEAPON_SLICE.stop)
        self.weapon_left, self.attack_left = INITIAL_WEAPON_STEPS, 0
        self.last_consumed = ([0] * len(DOOM_KEYS), 0.0)
        self.frame_id = 0
        self.jpeg = None
        # Small and lossy on purpose: a backlog is the growing-lag failure mode MJPEG had.
        self.queue = collections.deque(maxlen=8)
        self._lock = threading.Lock()

    def set_action(self, keys, turn, reset=False):
        self.keys = keys
        self.turn += turn
        if reset:
            self.reset_req = True

        for i in range(WEAPON_SLICE.start, WEAPON_SLICE.stop):
            if keys[i]:
                self.weapon_idx, self.weapon_left = i, WEAPON_HOLD_STEPS
                break
        if keys[DOOM_KEYS.index("attack")]:
            self.attack_left = max(self.attack_left, ATTACK_MIN_STEPS)

    def consume_keys(self):
        v = list(self.keys)
        v[WEAPON_SLICE] = [0] * (WEAPON_SLICE.stop - WEAPON_SLICE.start)
        if self.weapon_left > 0:
            v[self.weapon_idx] = 1
            self.weapon_left -= 1
        if self.attack_left > 0:
            v[DOOM_KEYS.index("attack")] = 1
            self.attack_left -= 1
        return v

    def consume_turn(self, limit: float = 12.0) -> float:
        with self._lock:
            turn = max(-limit, min(limit, self.turn))
            self.turn = 0.0
        return turn

    def latest(self):
        with self._lock:
            return self.frame_id, self.jpeg

    def publish(self, jpegs):
        """Publish one latent step's video frames (a list, chronological)."""
        if not isinstance(jpegs, (list, tuple)):
            jpegs = [jpegs]
        with self._lock:
            for jpeg in jpegs:
                self.frame_id += 1
                self.jpeg = jpeg
                self.queue.append((self.frame_id, jpeg))

    def drain(self):
        """Pop everything queued since the last call (streaming transports use this)."""
        with self._lock:
            out = list(self.queue)
            self.queue.clear()
            return out


class GPUWorker:

    def __init__(self, model, loader, pace_hz: float | None = None):
        import queue
        import threading

        self.model = model
        self._loader_iter = iter(loader)
        self.sessions: dict = {}
        self._create_q: queue.Queue = queue.Queue()
        self.alive = True
        self.ready = threading.Event()
        self.pace_hz = pace_hz
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def request_session(self, steps=N_DIFFUSION_STEPS) -> str:
        import uuid

        sid = uuid.uuid4().hex[:12]
        self._create_q.put((sid, steps))
        return sid

    def _make_player(self, steps=N_DIFFUSION_STEPS):
        seed_batch, _ = next(self._loader_iter)
        return InteractivePlayer(
            self.model, seed_batch,
            n_diffusion_steps=steps, noise_level=NOISE_LEVEL, device="cuda",
            guidance=GUIDANCE,
        )

    def _run(self):
        import io
        import time
        import traceback

        import torch
        from PIL import Image

        self._graphed = False
        try:
            if COMPILE_DECODE:
                self.model.decode_to_video = torch.compile(self.model.decode_to_video)
            t0 = time.time()
            self._player = self._make_player()  # the persistent player backing the graph
            if USE_CUDA_GRAPH:
                print("[play] capturing CUDA graph...", flush=True)
                self._player.setup_graph()
                self._graphed = True
            else:
                for _ in range(6 if COMPILE_DECODE else 2):
                    self._player.step([0] * len(DOOM_KEYS), 0.0)
            print(f"[play] warm in {time.time() - t0:.0f}s (graphed={self._graphed})", flush=True)
        except Exception:  # noqa: BLE001 -- fall back to eager rather than crash the container
            print("[play] graph/warmup failed (falling back to eager):\n" + traceback.format_exc(), flush=True)
            self._graphed = False
            if not hasattr(self, "_player"):
                self._player = self._make_player()
        self.ready.set()
        import gc

        gc.collect()
        gc.freeze()
        gc.disable()

        n, t0, cap = 0, time.time(), 6  # cap concurrent sessions
        pace_dt = 1.0 / self.pace_hz if self.pace_hz else 0.0
        next_deadline = time.perf_counter()
        while self.alive:
            while not self._create_q.empty():
                sid, steps = self._create_q.get()
                try:
                    if SINGLE_PLAYER:
                        self.sessions.clear()
                        try:
                            self._player.reseed_in_place(next(self._loader_iter)[0])
                        except Exception:  # noqa: BLE001 -- a stale scene beats no session at all
                            print("[play] reseed failed:\n" + traceback.format_exc(), flush=True)
                        self.sessions[sid] = Session(sid, self._player)
                    else:
                        if len(self.sessions) >= cap:
                            self.sessions.clear()
                        self.sessions[sid] = Session(sid, self._make_player(steps))
                    print(f"[play] session {sid} ready", flush=True)
                except Exception:  # noqa: BLE001
                    print("[play] session create failed:\n" + traceback.format_exc(), flush=True)

            if not self.sessions:
                time.sleep(0.005)
                continue

            for sess in list(self.sessions.values()):
                try:
                    if sess.reset_req:
                        sess.reset_req = False
                        sess.player.reseed_in_place(next(self._loader_iter)[0])
                    turn, keys = sess.consume_turn(), sess.consume_keys()
                    sess.last_consumed = (keys, turn)
                    if self._graphed:
                        jpeg = sess.player.step_fast(keys, turn)  # list of GPU-encoded bytes
                    else:
                        frames = sess.player.step(keys, turn)  # (T_video, H, W, 3)
                        jpeg = []
                        for f in frames:
                            buf = io.BytesIO()
                            Image.fromarray(f).save(buf, format="JPEG", quality=JPEG_QUALITY)
                            jpeg.append(buf.getvalue())
                    sess.publish(jpeg)
                except Exception:  # noqa: BLE001
                    print(f"[play] step failed {sess.sid}:\n{traceback.format_exc()}", flush=True)
                    self.sessions.pop(sess.sid, None)
                # Yield the GIL briefly so the async /step handlers aren't starved by the flat-out
                # generation loop -- keeps request latency low. Negligible cost vs the ~46ms step.
                time.sleep(0.001)
                n += 1
                if n % 30 == 0:
                    dt = time.time() - t0
                    print(f"[play] {30 / dt:.1f} gen fps ({dt / 30 * 1000:.0f} ms/frame)", flush=True)
                    t0 = time.time()

            if pace_dt:
                # Sleep out the remainder of this step's slot. Deadlines advance on a fixed grid so
                # a slow step is absorbed by the next one rather than shifting the clock; if we fall
                # more than one slot behind, resync instead of trying to catch up (which would
                # fast-forward the world to "make up time").
                next_deadline += pace_dt
                slack = next_deadline - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                elif slack < -pace_dt:
                    next_deadline = time.perf_counter()


def _load_model_and_seed(stage: str, dataset: str):
    """Load a WM checkpoint and one fresh context clip to seed a session."""
    from mira.data.training_loader import create_loader
    from mira.inference.loading import load_world_model

    ckpt = latest_checkpoint_path(run_dir(stage, dataset, ""))
    if not ckpt:
        raise RuntimeError(f"no {stage} checkpoint")
    model, _ = load_world_model(Path(ckpt), device="cuda")
    model.eval()

    ds = DATASETS[dataset]
    loader = create_loader(
        index_path=f"{ds['root']}/test",
        clip_len=model.config.n_context_frames + 4 * model.temporal_downsampling,
        target_fps=int(model.config.video.fps),
        batch_size=1,
        num_workers=2,
        frame_size=tuple(ds["frame_size"]) if ds.get("frame_size") else None,
        valid_keys=list(DOOM_KEYS),
        seed=int(time.time()) % 100000,  # a different starting scene each session
        infinite=True,
    )
    return model, loader, ckpt
