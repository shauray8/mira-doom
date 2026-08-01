# Playing MIRA in real time — deploy guide, internals, and drift control

`serve/play_app.py` serves the MIRA Doom world model as a **browser-playable, real-time game**:
you move with WASD, aim with the mouse, and the model *generates* every frame from your inputs. On
one H100 it generates at **~22–24 fps (42 ms/frame)** at 2 diffusion steps.

> On an RTX PRO 6000 Blackwell (sm_120), `serve/serve_local.py` runs at **35 fps with 6 diffusion
> steps**, real-time paced. See [§5b](#5b-the-blackwell-sm_120-optimisation-pass) for the full
> optimisation pass — what was profiled, what worked, and what regressed.

This document covers: what it is, how it works, how to run it, every
tuning knob, and — importantly — **how to fight drift** (the "world melts after a few seconds"
problem) at inference time, training-free.

---

## 1. What it is

- A single **GPU worker thread** holds the frozen world model + codec and a running KV-cache, and
  generates frames as fast as the GPU allows from the *current* action.
- A **FastAPI** app serves the browser page and a `/ws` WebSocket.
- The browser sends actions at 60 Hz on the same socket; the server **pushes** frames as they are
  generated into a small client-side jitter buffer, so displayed fps is independent of latency.

There is **no ground truth at play time** — after a short real-Doom seed (loaded once at startup),
everything you see is the model generating, conditioned on your controls.

---

## 2. Architecture (and why each piece exists)

```
browser ──WS {action} 60Hz──▶  FastAPI (async)  ──pushes JPEG as generated──▶  jitter buffer
                                        │
                                        ▼ (shared state: current action, latest JPEG)
                              GPU worker thread  (the ONLY GPU consumer)
                                        │  loop, ~42 ms/frame:
                                        │   1. encode the current action
                                        │   2. replay the CUDA graph (denoise ×N + decode)
                                        │   3. GPU JPEG-encode the frame
                                        └─▶ publish latest JPEG
```

The design was reached by fixing real bottlenecks in order — worth knowing so you don't undo them:

| Problem we hit | Fix in the code | Why |
|---|---|---|
| **~1 fps** | **Client-pull, not request-per-frame-with-generation** | The old design waited for a fresh generation on every HTTP request. Now the worker generates continuously and `/step` returns the *cached* latest frame instantly. |
| **FastAPI 422 on every request** | Removed `from __future__ import annotations` + parse the body via `Request` | Stringified annotations make FastAPI unable to resolve nested-closure endpoint types. |
| **torch.compile can't touch the denoiser** (`InternalTorchDynamoError`) | **Manual `torch.cuda.CUDAGraph`** capture of the step | The denoiser has data-dependent control flow (the repo docs warn about this); a hand-captured graph records the kernel stream directly and bypasses tracing. This is the 2× lever to 24 fps. |
| **Cross-thread CUDA-graph crash** | **Single GPU worker thread** does warmup, capture, *and* every step | CUDA graphs can't be replayed on a different thread than they were captured on. |
| **Growing lag over time** | **Client-pull + pipeline** instead of MJPEG push | MJPEG streamed every frame; when the client consumed slower than the server produced, frames backed up in the TCP buffer and the browser always showed a *stale* frame (lag grew). Pull keeps only fresh frames in flight. |
| **Input registered inconsistently** | **document-level** mouse listeners + hold-to-fire | During pointer-lock, button events go to the document, not the canvas; the weapon also cycles at Doom's own rate, so fire must be *held*. |
| **Intermittent drops to ~3 fps** | **`gc.freeze()`** after warmup | Python's GC periodically pauses the hot loop; freezing moves existing objects out of its reach. |
| **CPU JPEG + 590 KB frame copy per step** | **GPU nvJPEG** (`torchvision.io.encode_jpeg` on CUDA) | Encodes on the GPU and copies only ~30 KB of compressed bytes — off the GIL, off the critical path. |
| **Trans-Pacific latency** | **`region="ap"`** (deploy near the player) | Once generation is fast, the browser↔server round-trip dominates. Run the container near the user. |

### The CUDA-graph reimplementation (the subtle part)

`LatentWorldModel.denoise_streaming` reallocates the KV-cache (`torch.cat`) every step — fatal for a
CUDA graph, which needs **fixed memory addresses**. `InteractivePlayer` reimplements the steady-state
step (`_denoise_body`) so that:

1. `z_t` and the KV-cache live in **fixed buffers**, updated **in-place** (roll + copy, no realloc).
2. All randomness comes from **static noise buffers filled outside the graph** — otherwise the graph
   replays identical noise and the picture freezes.
3. Only the **steady-state** path is captured (the cache is pre-populated by a short warmup, so the
   first-call "build the cache" branch never runs inside the graph).
4. `None` cache slots (only every `time_attention_every`-th layer caches) and the bf16/fp32 autocast
   boundary at the decoder are handled explicitly.

If capture fails for any reason it **falls back to eager** (still correct, ~2× slower) rather than
crashing — see `GPUWorker._run`.

---

## 3. Configuration knobs

All at the top of `play_app.py`, and overridden from the environment by `serve_local.py`:

| Knob | Default | Effect |
|---|---|---|
| `PLAY_STAGE` | `"wm"` | Which checkpoint to play (`wm` = single-player base; `wm_long` = long-horizon finetune). |
| `N_DIFFUSION_STEPS` | `2` | Denoising steps/frame. **2** = fastest/softest, **3–4** = cleaner + slightly less drift, slower. Set with `MIRA_STEPS`. |
| `NOISE_LEVEL` | `0.45` | **The main drift knob** — see §5. Higher = more stable, softer. |
| `USE_CUDA_GRAPH` | `True` | The 24 fps path. `False` → eager (~11 fps) but supports live reseed/reset. |
| `SINGLE_PLAYER` | `True` | Required by the CUDA graph (one persistent player = fixed buffers). |
| `JPEG_QUALITY` | `72` | Lower = smaller/faster transfer, blockier. |
| `PIPELINE` (client JS) | `4` | Concurrent pull requests; raise for jittery/high-latency networks. |
| `region` (`@app.cls`) | `"ap"` | Deploy near the player. `us-east|us-west|eu|ap|jp|au|ap-south…` |
| `scaledown_window` | `120` | Seconds idle before the GPU spins down (0 cost while idle). |

---

## 4. Running it

```bash
./setup_inference.sh    # deps + checkpoints + seed clips (one time)
./infer.sh              # -> http://<box>:3754
```

`infer.sh` runs `serve/serve_local.py`, which loads the newest checkpoint under `ckpts/doom_<stage>`,
starts one paced `GPUWorker`, and serves the browser client. What it needs on local disk:

- `ckpts/doom_wm/checkpoint-*/checkpoint.pth` + `world_model_config.yaml`
- `ckpts/doom_codec/checkpoint-*/checkpoint.pth` (the codec also ships frozen inside the WM
  checkpoint, but the WM config points at this path — `serve/config.py::ensure_codec_path` symlinks
  `/runs/doom_codec` at it, because that absolute path is baked into the checkpoint's own config and
  cannot be overridden)
- `data/doom_mira/test` for the seed clip

Env that matters: `MIRA_ATTN_BACKEND=sdpa` is set for you (the KV-cache path uses SDPA regardless —
FA3 does **not** accelerate single-frame streaming, only full-sequence training). `torchvision` must
be a build with **nvJPEG** for GPU encode; otherwise `encode_jpeg` auto-falls back to PIL.

Cold start = weights load + decoder compile + **CUDA-graph capture** (~2 min cold, ~20 s once the
inductor cache is warm). Keep the process warm; don't recapture per request.

---

## 5. Drift — why it happens and how to fight it (training-free)

**Why it melts:** the model is autoregressive — every frame conditions on its *own* previous outputs
through the KV-cache. Small errors compound: a slightly-off frame becomes the context for the next
one, so error grows and after ~2–4 s the world wanders off-distribution. We measured this: per-frame
PSNR vs a reference fell ~30 → ~18 dB over 60 generated frames.

You cannot retrain here, but there is a lot you can do **at inference**, in rough order of
payoff/ease:

### 5.1 Context noise — `NOISE_LEVEL` (the #1 free knob, already on)
The model was trained with **diffusion forcing** (independent noise per context frame), so it's
robust to *noisy* context. Injecting noise into the context at inference makes it ignore fine detail
that's probably erroneous, which slows compounding. We swept it: **0.45** flattened the drift slope
~25 % and lifted the far-horizon PSNR ~1.5 dB vs the 0.2 default, at no cost. Try **0.4–0.55**.
Tradeoff: higher = more stable but softer / less fine detail.
*In CUDA-graph mode this is baked into the captured graph — change the constant and redeploy; it
can't be tuned live.*

### 5.2 Periodic **hard reset** (what you've seen others do)
Every N seconds, clear the KV-cache and reseed from a real dataset clip. Instantly kills *all*
accumulated drift. Downside: the scene **jumps** to a new location (a visible discontinuity) — fine
as a "new scene" button, jarring for continuous play. This is `InteractivePlayer.reset()`; it's
currently disabled in single-player CUDA-graph mode because reseeding reallocates buffers and
invalidates the graph (you'd recapture, ~30 s). Two ways to use it:
- Run with `USE_CUDA_GRAPH=False` (eager, ~11 fps) where `reset()` works freely, **or**
- Keep the graph and, on reset, tear down and re-run `setup_graph()` (accept a ~30 s pause).

### 5.3 Periodic **soft reset / re-anchoring** (the nicer version)
Instead of reseeding to a *different* real clip, re-ground the model to its **own current frame**:
decode the latest latent → pixels → **re-encode** through the frozen codec → use that as fresh clean
context. Because the codec is deterministic, this snaps the latent back onto the valid latent
manifold — removing latent-space error that has drifted off-distribution — **without changing the
visible scene**. It's the closest thing to "restart without restarting."
Sketch (every ~1–2 s, eager / graph paused):
```python
frame = model.decode_to_video(player.z_t[:, -1:])          # current generated pixels
clean = model.encode_video(frame_as_batch)                 # re-encode -> on-manifold latent
# rebuild z_t context + KV-cache from `clean` (same as _seed but from the generated frame)
```
Cost: one decode+encode round-trip per anchor. In graph mode, do it as an occasional pause or run
eager. This is the highest-quality anti-drift option for continuous Doom.

### 5.4 Shorter context window
The KV-cache is a sliding window (~20 latent frames here). A **shorter** window pushes old (drifted)
frames out of context sooner, so errors leave the horizon faster — at the cost of the world
"forgetting" more quickly (objects behind you may not persist). Tunable in the streaming loop.

### 5.5 More diffusion steps
`N_DIFFUSION_STEPS=2` is the fast floor; **3–4** produce cleaner per-frame latents, so there's less
error to accumulate. Costs fps. `MIRA_STEPS=3` to feel it.

### 5.6 Smooth your inputs
Rapid action changes push the model to *extrapolate*, which accelerates drift. Steady WASD + gentle
mouse holds together far longer than frantic input.

**Practical Doom recipe:** `NOISE_LEVEL ≈ 0.45–0.5`, `N_DIFFUSION_STEPS=3`, and a **soft re-anchor
(§5.3) every ~2 s** (or a hard reset button if you're OK with the scene jump). That combination gives
the longest coherent sessions without retraining.

---

## 5b. The Blackwell (sm_120) optimisation pass

`serve/serve_local.py` is the entry point: it loads a checkpoint, starts one paced `GPUWorker`, and
serves the browser client. The engine (`InteractivePlayer`, `Session`, `GPUWorker`, `encode_jpeg`)
lives in `serve/play_app.py`, so the model, CUDA-graph and JPEG paths are exactly the code above.

```bash
python serve/serve_local.py            # port 3754
MIRA_STAGE=wm_long MIRA_CTX=78 MIRA_STEPS=4 python serve/serve_local.py
```

All numbers below are measured on one **RTX PRO 6000 Blackwell Server Edition (sm_120, 97 GB)**,
torch 2.8.0+cu128, at 384×512. Note that sm_120 is the *consumer/workstation* Blackwell, not the
sm_100 datacentre part.

### Where the frame budget actually went

Profiling the graphed step first, before changing anything, contradicted the obvious guesses — it is
**not** attention kernels and **not** the world-model transformer:

| Component | Cost | Share of a 47.3 ms frame |
|---|---|---|
| codec ViT decoder (`decode_to_video`) | 26.0 ms | **55 %** |
| DiT forward × (n_diffusion_steps + 1 renoise) | 6.2 ms each | ~40 % |
| action encode | 0.4 ms | 1 % |

And inside the decoder, one kernel dominated: `fmha_cutlassF_bf16_aligned_64x128_rf_sm80`, **28 calls
per frame at 372 µs each = 10.4 ms**. Twenty-eight is exactly the decoder's `vit_depth`.

### 1. A provable no-op costing 22 % of the frame (exact)

The codec decoder runs **causal temporal self-attention over the frame axis**. Interactive play
decodes *one latent frame at a time*, so that attention runs with `seq_len == 1` in all 28 blocks —
and a softmax over exactly one key is 1, making the output identically `v`:

```python
>>> out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)   # S = 1
>>> (out - v).abs().max().item()
0.0                       # bit-exact, not approximate
```

It was not free, though: at (768 batch, 16 heads, seq 1, head_dim 72) the cutlass mem-efficient
kernel tiles that degenerate shape terribly — **375 µs**, against **3.8 µs** to copy `v`. `attend()`
now short-circuits `q_len == k_len == 1` (broadcasting kv heads for GQA). Verified bit-identical
against the SDPA reference across head counts, GQA ratios and both causal settings.

> **decode 30.0 → 19.4 ms · frame 47.3 → 37.0 ms (21.2 → 27.0 fps)**

### 2. A causal mask that is entirely `True` (exact)

`_sdpa()` materialises `local_causal_mask(...)` whenever `causal=True`. In incremental decoding
(`q_len == 1`, unbounded `context`) that mask is **all-`True`** — semantically a no-op:

```python
>>> local_causal_mask(q_len=1, k_len=40, context=None, device=dev).all()
tensor(True)
```

But passing *any* explicit `attn_mask` disqualifies SDPA's flash and cuDNN backends and forces the
slow cutlass path. That is the shape every streaming step uses, on all 5 of the DiT's
temporal-attention layers, on every diffusion step. Skipping it when the mask is vacuous (a bounded
`context` window is a real mask and still gets built) agrees with the masked reference to one bf16
ULP — 0.0078, the same tolerance §FA3 accepts for its own kernel check.

> **≈ 6 ms/step**

### 3. Compiling the codec decoder

The decoder's remaining cost was thousands of tiny elementwise kernels (LayerNorm, LayerScale,
SwiGLU, RoPE, QK-norm) — ~1670 launches per decode at 2–4 µs each. Inductor fuses them:

```python
model.codec.decoder = torch.compile(
    model.codec.decoder, mode="max-autotune-no-cudagraphs", dynamic=False)
```

`-no-cudagraphs` is **required**: `InteractivePlayer.setup_graph` captures its own graph around the
whole step, and inductor's cudagraph wrapper fights it. Max abs error vs eager is 0.0039 (bf16
rounding on pixels in [0, 1]). Costs ~55 s of compile at startup.

> **decode 19.4 → 8.6 ms · frame → 28.7 ms (34.9 fps)**

### 4. Do NOT compile the DiT (measured regression)

Tempting, and it does help the eager path — but under the manual CUDA graph it is **slower**:

| | eager | graphed |
|---|---|---|
| DiT not compiled | 48.4 ms | **28.7 ms** |
| DiT compiled | 35.5 ms | 31.9 ms ❌ |

The graph already eliminates the launch overhead compilation was buying, and inductor's Triton GEMMs
lose to cuBLAS at these shapes (192 tokens, hidden 2048). Left uncompiled deliberately.

### 5. The decoder was producing two frames and one was thrown away

The codec has `patch_size_t=2`, so **one latent step decodes two consecutive video frames**:
`decode_to_video` returns `(1, 2, 3, 384, 512)`. The original loop kept `frame[0, -1]` and discarded
the other — the decoder had already paid for both. Publishing both doubles the delivered frame rate
for nothing.

### 6. Real time is 17.5 steps/s, not "as fast as possible"

Because each latent step advances `temporal_downsampling = 2` video frames of a **35 fps** game,
real time is `video.fps / temporal_downsampling = 17.5` latent steps/s — a **57.1 ms** budget per
step. Generating faster does not make a better game, it runs the world in fast-forward. `GPUWorker`
now paces to that grid (absorbing slow steps into the next slot, resyncing rather than sprinting if
it falls more than a slot behind), and the surplus speed is spent on **denoising steps instead of
frame rate**:

| diffusion steps | wm (19-latent ctx) | wm_long (39-latent ctx) |
|---|---|---|
| 2 | — | 28.1 ms (2.03×) |
| 3 | — | 34.5 ms (1.66×) |
| 4 | — | 40.8 ms (1.40×) |
| 5 | 45.5 ms (1.26×) | 46.5 ms (1.23×) |
| **6** | **51.8 ms (1.10×)** | 52.2 ms (1.09×) |
| 7 | 57.3 ms (1.00×) | — |
| 8 | 62.5 ms (0.91×) ❌ | — |

Default is **6** — the last setting comfortably inside the budget, up from the original default of 2.
7 lands exactly on it with no slack for JPEG encode, socket writes or a reseed hitch, and 8 is both
too slow *and* lower quality (§6c).

### 6b. `NOISE_LEVEL` depends on the step count — 0.45 does not transfer

§5.1's 0.45 was swept on the **2-step** configuration, where context noise is damage control for
noisy latents. At 6 steps the latents are clean enough that added context noise only destroys
detail, and the optimum collapses to **0**.

Judged by **HUD survival**: the status bar is a fixed, high-contrast structure the model renders
crisply while on-distribution and smears as it drifts, so "steps until the HUD band decorrelates
from its initial appearance" measures drift resistance directly. (Sharpness does not work — it
happily rewards high-frequency drift artefacts. A first sweep ranked a fully-melted frame top.)

Over 3 seed clips × 2 RNG seeds × 150 steps at 6 diffusion steps:

| `noise_level` | mean HUD@150 | worst run | runs that never broke |
|---|---|---|---|
| **0.00** | **0.667** | **0.396** | **3/6** |
| 0.15 | 0.540 / 0.133 (two sweeps) | −0.282 | 3/6, then 0/6 |
| 0.20 | 0.255 | −0.327 | 0/6 |
| 0.30 | 0.044 | −0.120 | 0/6 |
| 0.45 *(old default)* | 0.026 | −0.541 | 0/6 |

0.0 is also what the checkpoint's own offline metrics use
(`world_model_metrics.inference.noise_level: 0.0`).

> **These rollouts are chaotic and NOT bit-reproducible** — cuBLAS split-k reductions are
> non-deterministic and autoregression amplifies any epsilon. The same config scored 150 and 36 on
> consecutive single runs. Never tune this from one rollout; the table above is 6 runs per row and
> the spread is still visible.

### 6c. More diffusion steps is not monotonically better

8 steps is **worse** than 6, not just slower — and it also misses the real-time budget (61.9 ms vs
57.1, ~0.92×, i.e. 9 % slow motion):

| | HUD@50 | HUD@100 | HUD@150 | steps alive |
|---|---|---|---|---|
| 6 steps, noise 0.30 | 0.950 | 0.946 | 0.946 | 150/150 |
| 8 steps, noise 0.30 | 0.857 | 0.439 | 0.457 | 43/150 ❌ |
| 8 steps, noise 0.45 | 0.769 | 0.687 | 0.693 | 150/150 |

6 is both the fastest setting that fits real time *and* the best-scoring one, so there is no
quality/speed trade to make above it.

### 7. Transport: client-pull was costing a distant player 2/3 of the frames

§2's client-pull-over-HTTP was a workaround for a hosted-platform WebSocket bug. That constraint is gone,
and pull costs **one network round trip per frame**: a server generating 21 fps delivered ~7 fps to a
far-away browser, because throughput is `PIPELINE / RTT`. `serve_local.py` pushes frames over a
WebSocket into a small client-side jitter buffer (start at 3 frames, drop oldest above 10) and draws
on a 35 fps clock, so **displayed frame rate is independent of latency**.

> **35.2 fps measured, identical on localhost and through the RunPod proxy · ~22 KB/frame,
> 6.4 Mbit/s at JPEG quality 78**

### 8. Reset works under the CUDA graph now (`reseed_in_place`)

§5.2 says reset is disabled in graph mode because reseeding reallocates buffers. It does not have to:
reseed through the normal path onto temporaries, then **copy the values into the graph's existing
buffers** and restore the original tensor objects. Shapes are fixed by construction, so the graph
stays valid. Verified: `id(z_t)` and every `id(k)` unchanged, and speed unchanged after reseed.

This also fixes a real defect. With `SINGLE_PLAYER` the one persistent player generates continuously
from container start, so a browser connecting ten minutes later **inherited ten minutes of drift** —
an already-melted world. Sessions now reseed from a fresh clip on connect, and `R` reseeds on demand.

### 9. Input handling matched to the training distribution

Measured over 274 s of the training split (per-key run lengths), and it contradicted the client:

| key | on-rate | mean hold | median | max |
|---|---|---|---|---|
| `weapon1..7` | 0.5–39 % | 2.2–4.1 f | **2 f** | 6–56 f |
| `attack` | 31.5 % | 6.3 f | 2 f | 78 f |
| `speed` (run) | **87.9 %** | 19.1 f | 10 f | 174 f |

- **Exactly one weapon is ever set at a time** (>1 simultaneous: 0.00 % of frames). The old client
  latched a weapon bit on *forever* once pressed — far outside that distribution, which is why the
  weapon would not come out. Latching now happens server-side and is capped at one weapon.
- Holds are counted in **latent steps, not client frames**. The browser samples input at ~60 Hz while
  the model consumes 17.5 actions/s, so a client-side "hold for 8 frames" had no fixed duration in
  model time. `WEAPON_HOLD_STEPS = 2`, `ATTACK_MIN_STEPS = 3`, `INITIAL_WEAPON_STEPS = 6`.
- **Mouse turn accumulates instead of overwriting.** `Session.set_action` summed at 60 Hz and is
  drained once per step, so a flick delivers all of its rotation; previously ~2/3 of every mouse
  movement was overwritten before the model ever saw it. Sensitivity also dropped 0.6 → 0.22 (live
  slider), because at 0.6 any real drag saturated the ±12-per-step clamp.
- **`speed` defaults on**, since running is the model's normal state (87.9 %).
- The on-screen input panel echoes the action the **server consumed**, not what the browser sent —
  the two differ by exactly these latches.

### 10. Memory horizon (optional, `wm_long` only)

`set_inference_context(n)` is an inference-time rollout knob capped by the trained latent window
`video.timesteps`. `wm_long` was trained at `timesteps=80`, so it accepts a 40-latent window:

> **1.09 s → 2.23 s of gameplay memory for +3 ms/step**

This is the fix for "look at the floor, look up, and the world has teleported". The base `wm` was
trained at `timesteps=40` and **cannot** be stretched — 38 and 39 context frames both floor to 19
latents. Enable with `MIRA_STAGE=wm_long MIRA_CTX=78`.

It is **not** the default, because `wm_long` is only a 13k-step long-horizon finetune on top of the
75k-step `wm` and it visibly traded action adherence for drift stability: in play it fires without
input and ignores weapon select. `wm` keeps control and gives up the longer memory.

### 11. Negative result: action classifier-free guidance

`dropout_action_prob=0.1` means the model has a genuine unconditional mode, and the null-action
embedding can be reconstructed exactly — the dropout tokens are substituted after temporal pooling,
so `joint_mlp([mouse_dropout_token, keyboard_dropout_token])` reproduces it. Implemented behind
`MIRA_GUIDANCE` (extra DiT forward per step; 3 guided steps ≈ the cost of 5 unguided).

**It is off by default because it measurably degrades output on this checkpoint**: at 1.5 the HUD
smears and the scene goes blocky; at 2.5 the image collapses entirely. The unconditional mode is too
weakly trained on a 13k-step finetune to serve as a CFG negative. Do not enable without re-measuring.

### 12. Attention backends on sm_120

`MIRA_ATTN_BACKEND=sdpa`, and nothing is lost by it. FA3 is a **Hopper-only** build and will not
compile for sm_120 — but more importantly `mira.ml.attention` deliberately keeps the *incremental
decode* path (KV-cache, `q_len=1`) on SDPA regardless of backend, and that is the only path
interactive play uses. FA3/FA2/SageAttention accelerate full-sequence training attention, which does
not run here. The wins above came from removing attention work, not from faster attention kernels.

**§5c profiles this claim on a 5090 and quantifies it**: all attention together is 3.2 ms of a
58.2 ms frame, the shapes are 192-token spatial and 1×20 decode, and the codec decoder's attention
has `head_dim=72` — a size SageAttention does not support at all. Read that before trying Sage-3.

### Net effect

| | before | after |
|---|---|---|
| step time (equal settings) | 47.3 ms | **28.7 ms** (1.65×) |
| diffusion steps per frame | 2 | **6** |
| video frames delivered per step | 1 of 2 | **2 of 2** |
| displayed frame rate, local | ~21 fps | **35 fps** (real-time paced) |
| displayed frame rate, distant browser | ~7 fps | **35 fps** |
| VRAM | 11.5 GB | 19.5 GB |

## 5c. A second bare-GPU deployment: GeForce RTX 5090 (32 GB), from an empty checkout

§5b was measured on an RTX PRO 6000 Blackwell (97 GB). This section is a second, independent
bring-up of the same `serve_local.py` on a **GeForce RTX 5090** (sm_120, 32 GB, RunPod, torch
2.8.0+cu128, driver 580, CUDA 13 runtime) starting from a checkout with **no `ckpts/` and no
`data/`**. Everything below is measured on that box.

### What the box actually needed

| Step | Command / fact | Gotcha |
|---|---|---|
| Python deps | `pip install -e '.[hydra,decode,viz]' fastapi 'uvicorn[standard]'` | — |
| Video decode | **`pip install torchcodec==0.7.0`** | The current wheel (0.15) links `libnvrtc.so.13` — a **CUDA 13** build. With torch 2.8+cu128 it fails to load, and `preprocess_doom.py` catches the exception per episode and cheerfully writes `index.json` with **0 matches**. Silent, not a crash. |
| DINOv3 backbone | `pip install torchmetrics termcolor` | `VideoCodec.load_from_checkpoint` builds the DINOv3 architecture through `torch.hub` even with `pretrained=False` (weights come from the codec checkpoint). The hub repo's `hubconf.py` imports `dinov3.eval.segmentation` → `torchmetrics` at module scope, so without it **model loading dies**, deep inside `LatentWorldModel.__init__`. `pyproject` lists these only as transitive `[train]` deps. |
| Weights | `ckpts/doom_{codec,wm,wm_long}/checkpoint-*/checkpoint.pth` + the run's config YAML | Layout must match `run_dir(stage, "doom")`. `_ensure_codec_path()` then symlinks `/runs/doom_codec` → there, satisfying the absolute `codec_checkpoint` baked into `world_model_config.yaml`. |
| Seed clips | 2 shards (1.1 GB) of `chrisxx/doom-2players-mp4`, then `preprocess_doom.py --val-fraction 1.0 --max-episodes 4` | The repo ships no `data/`, and the play path **cannot start without it** — `_load_model_and_seed` needs `doom_mira/test` for the context clip. 4 episodes → 4 matches × 39 chunks × 2 views ≈ **312 distinct starting scenes**, 975 MB. Budget ~12 GB of host RAM: `decode_all_frames` holds an entire episode uncompressed. |

`--val-fraction 1.0` routes every episode to `test/`, which is the only split the play path reads.

### Measured

| | RTX 5090 (this box) | RTX PRO 6000 (§5b) |
|---|---|---|
| graphed step, 6 diffusion steps, `wm` | **58.2 ms** | 51.8 ms |
| real-time budget (`td`/`video.fps`) | 57.1 ms | 57.1 ms |
| headroom | **−1.0 ms (just over the line)** | +5.3 ms |
| delivered over the WebSocket | **35.1 fps** | 35 fps |
| `wm_long` + `MIRA_CTX=78` (2.23 s memory) | **33.2 fps** | — |
| VRAM, `wm` | 15.7 GB of 32 GB | 19.5 GB |
| cold start | 111 s first run, **19 s** with a warm inductor cache | ~30 s |

The 5090 lands ~1 ms *over* the real-time budget at 6 steps, so it paces at essentially exactly
35 fps with no slack. `MIRA_STEPS=5` buys back ~8 ms if you want margin.

### The SageAttention-3 question, answered with a profile

Worth writing down because the intuition ("Blackwell, FP4 attention, 5× faster kernels") is
appealing and **wrong for this workload**. §12 said attention is not the bottleneck; here is the
measurement on the 5090, per latent step at 6 diffusion steps:

| Attention site | Shape | µs/call | ms/step | head_dim | Sage-eligible? |
|---|---|---|---|---|---|
| DiT spatial (16 layers × 7 passes) | q(1, **192**, 16, 128) kv(1, 192, 4, 128) | 26.1 | 2.93 | 128 | dim OK, but S=192 |
| DiT temporal (5 caching layers × 7) | q(192, **1**, 16, 128) kv(192, **20**, 4, 128) | 26.7 | 0.93 | 128 | **no** — `q_len=1` decode |
| codec decoder spatial (28 blocks) | q(1, 768, 16, **72**) | 39.0 | 1.09 | **72** | **no** — Sage supports 64/128 |

- Profiler-measured device time for **all** attention: **3.2 ms of a 58.2 ms frame (5.5 %)**;
  per-call wall-clock upper bound 4.95 ms (8.5 %).
- The latent grid is **12×16 = 192 tokens**. A 192×192 attention and a 1×20 decode cost the *same*
  26 µs — the signature of pure launch overhead, not compute. The DiT spatial attention runs at
  ~0.3 % of the 5090's bf16 peak, and the CUDA graph has already removed the launch overhead.
- Ceiling on the whole idea: if attention were **free**, 58.2 → 53.2 ms, i.e. 34.4 → 37.6 fps. And
  more fps is not wanted — the worker paces to real time on purpose (§6 of 5b).
- Packaging cost: SageAttention-3's FP4 kernels ship as **cu130 / torch 2.10–2.11** wheels. Adopting
  them means moving torch, which invalidates the `torchcodec==0.7.0` pin and the autotune cache.

**Verdict: do not.** Where the time actually goes is GEMMs (~56 % of CUDA time) plus copies and
elementwise work. If you want FP4 on a 5090, quantise the DiT's **linear layers**, not its attention.

## 5d. WorldKV — episodic memory for a model that forgets (implemented, negative result)

`serve/worldkv.py` + `serve/bench_worldkv.py` + `serve/sweep_worldkv.py`. **Off by
default** (`MIRA_WORLDKV=1` to enable). It is documented here because it is a clean, cheap,
training-free implementation that *did not work on this model*, and the reasons are specific and
worth knowing before anyone tries it again.

### What it is

From *WorldKV: Efficient World Memory with World Retrieval and Compression* (Yi et al.,
arXiv:2605.22718), as deployed at Reactor and described in Avik Sethia's write-up. The KV of frames
evicted from the attention window is not freed but **archived in a large bank**, keyed by the camera
pose it was generated from. Each window advance, the bank is queried for archived frames whose pose
is nearest the current one, and the best matches are written into a small **reserved slice** of the
attention window, with their positional timestamps rewritten so the model reads them as nearby
context. The model is untouched: no retraining, no new weights, same window size, same compiled
graphs. Claimed cost ~1 %. It is *episodic* memory bolted onto unchanged *semantic* memory.

### Why this codebase is an unusually good host

Three properties, all verified here:

1. **Positions are assigned by slot, not baked into the cache.** `SelfAttention.forward`
   concatenates the cache and only *then* applies RoPE (`k = cat([k_ctx, k]); apply_rotary_emb(k)`),
   so archived keys are stored **un-rotated** and a retrieved frame simply adopts the position of the
   slot it lands in. The paper's "rewrite the positional timestamps" step is **free** here.
2. **The cache is already fixed-address buffers** that `setup_graph` mutates in place, so a bank
   allocated *before* capture can be read and written between replays.
3. **No change to the captured step is needed.** The in-graph slide is
   `k_buf[:, :-1] = k_buf[:, 1:]; k_buf[:, -1:] = new`. Overwriting slots `[0, R)` *after* each
   replay has exactly the right semantics — the shift carries the oldest working frame into slot
   `R-1`, where it is overwritten (i.e. evicted). `play_app.py` stays verbatim; `worldkv.py` only
   wraps methods.

### The cue: there is no camera pose, so yaw is dead-reckoned

MIRA-Doom is conditioned on **actions, not pose**, so there is no 4×4 matrix to key on. `turn` is a
direct scalar input, so yaw can be integrated from it — but the unit was unknown. Calibrated against
**ground-truth video**, not guessed:

- Doom renders a **90° horizontal FOV**, so a full revolution scrolls the image by exactly 4 screen
  widths (2048 px at 512 wide). Estimating the per-frame horizontal shift by 1-D cross-correlation on
  pure-rotation frames (no strafe, which also shifts horizontally) and regressing against `turn`:
  **5.731 px per turn-unit, r = 0.966 → 1 turn-unit = 1.0075°, one revolution = 357.3 ≈ 360 units.**
- So `turn` is simply **degrees**. The recorded values are quantised to 1.25 and clamped to ±12.5.
  (A plausible-looking alternative — Doom's 16-bit BAM angle scaled by 1/102.4, which would have put
  a revolution at 640 units — is wrong. Measuring took ten minutes and saved a silent 1.8× error.)
- A first attempt to calibrate by correlating whole frames against Δcumulative-turn produced a
  **flat curve with no peak**: the player translates while spinning, so equal yaw does not mean equal
  view. Consecutive-frame optical shift removes that confound.

**Position is deliberately not dead-reckoned.** With no collision feedback the estimate walks through
walls and drifts without bound. This remembers *facings*, not places.

### What was built

- **Bank**: a ring indexed by `step % capacity`. Only the 5 of 16 layers with temporal attention
  cache (`time_attention_every=4` → layers 0, 4, 8, 12, 15), each `(192, 19, 4, 128)` bf16, so one
  archived latent frame costs **1.97 MB**. Measured capacities: **1 GB = 509 frames = 29 s**,
  **8 GB = 4069 frames = 232 s** of episodic memory. Allocated before graph capture, never resized.
- **Retrieval**: circular yaw distance over the metadata array only — the KV tensors are never
  searched, only moved when chosen. Retrieves a **contiguous run** of R frames ending at the best
  match (see "what did not work"), placed chronologically at the oldest slots.
- **Exact-baseline fallback**: slots not filled by retrieval are refilled with the frames that would
  have occupied them anyway (`t-(W-1)+s`), which are in the bank because every generated frame is
  archived. **Asserted bit-identical** across all layers and slots, so enabling WorldKV with no
  matching memory is a no-op, which is what makes the A/B trustworthy.
- **Harness**: `bench_worldkv.py` runs the article's dumpster test — swing away, dwell, swing back —
  against three conditions on the same clip and RNG seed: `still` (no rotation: the ceiling, pure
  drift), `off`, `on`. Score is HUD-cropped grayscale correlation between the view before and after.
  Dwell auto-scales to the model's window so retrieval can actually engage.

### What worked

- **The mechanism.** Frames are archived and retrieved correctly under both the eager and the
  **CUDA-graph** path, at 33–35 fps, with no corruption and no graph recapture.
- **The cue is accurate.** Mean |Δyaw| at retrieval was **0.0°** — dead reckoning off `turn` matches
  exactly. This is *not* a cue-quality failure.
- **The cost is negligible**, as claimed: scoring is a few thousand float32 on the CPU, and the copy
  is ~2 MB per archived frame plus R×1.97 MB per step of device-to-device traffic.
- **The fallback is exact**, verified by assertion, so the feature is free when it does not fire.

### What did not work — the result

Sweep: 8 clips × 2 RNG seeds, paired, `awayback` protocol (2.29 s away vs a 1.09 s window), `wm`:

| condition | mean return-consistency | Δ vs off | paired wins | paired sd |
|---|---|---|---|---|
| off (baseline) | +0.476 | — | — | — |
| on, R=2 of 19 slots | +0.495 | +0.019 | 10/16 | 0.042 |
| on, R=4 | +0.491 | +0.015 | 8/16 | 0.036 |
| on, R=8 | +0.474 | −0.001 | 8/16 | 0.092 |

The best case is ~1.8σ — a coin flip. In a separate 6-clip run the sign **flipped** (R=4 scored
−0.065 vs off), which is the definition of no effect. For scale, the `still` ceiling in that run was
+0.68 against off +0.54, so there was real headroom and WorldKV recovered none of it. Memory was
pinned in-window on 31 % of steps, so it had its chance.

**Why it fails, mechanistically:**

- **The window is too short for the trade to pay.** Retrieved frames land at the *oldest* RoPE
  positions, where 15+ contiguous recent frames outvote them — and every reserved slot is taken from
  working memory 1:1. The monotone decay from R=2 → R=8 (+0.019 → −0.001) is exactly that signature:
  more memory, more damage to continuity, net zero. WorldKV's setting is an **8 s** window where
  retrieved chunks displace a much smaller proportion of context, in models trained to use distant
  positions for long-range consistency. MIRA-Doom at **1.09 s** was not.
- **Doom-specific hazard** (anticipated, never reached): the status bar is in-frame and time-varying,
  so retrieved context carries a stale HUD that fights the current one. The open-world footage the
  method was demonstrated on has no HUD.

### Bugs found on the way (all in the harness, not in `play_app`)

1. **`codec.preprocess_batch` mutates its argument** (`batch.video = batch.video / 255.0`). Seeding a
   second player from the same clip divides an already-normalised video by 255 **again** and the run
   is black mush from frame one. This invalidated an entire A/B run before it was spotted — the
   `on` condition looked catastrophically broken when it was simply the third player built from one
   batch. **The server never hits this** (it pulls a fresh clip per player); any harness that pins a
   clip across conditions must `deepcopy`.
2. **Temporal non-maximum suppression suppressed exactly the right candidates.** The frames matching
   a given yaw are *consecutive* frames of one sweep, so an 8-step spread constraint rejected all but
   one and filled 0.08 of 4 slots. Contiguous-chunk retrieval fixed it (and is closer to the paper,
   which archives half-second chunks, not instants).
3. **Caching the write plan froze the fallback slots.** The retrieval *decision* is sticky for
   `refresh` steps, but the baseline fill tracks `t` and must be recomputed every step or the two
   oldest context frames stall — silently breaking the exact-baseline property.
4. **Stats counted decision events, not steps with memory pinned**, understating coverage 4×
   (`refresh=4`). "8 % of steps" was really 31 %.

### What was not measured

The quantitative sweep was **never run on `wm_long`**. It is the checkpoint most likely to benefit —
2.23 s window with `MIRA_CTX=78`, and finetuned at `noise_level=0.45`, i.e. explicitly trained to
tolerate corrupted context, which is what a retrieved jump-cut looks like. It was served and judged
by eye only (`MIRA_STAGE=wm_long MIRA_CTX=78 MIRA_WORLDKV=1 MIRA_WORLDKV_SLOTS=4`, 33.2 fps, 4 GB
bank = 116 s of memory) and showed no obvious improvement. If anyone revisits this, that sweep is
the first thing to run.

## 6. Known limitations

- **Single player** in CUDA-graph mode (one persistent player backs the graph). Multiple browsers
  share one world. For true multi-session, run eager (`USE_CUDA_GRAPH=False`) — ~11 fps but each
  session gets its own player.
- **Reset disabled** in graph mode (§5.2).
- **Cold start ~30 s** (graph capture) on the first request after idle.
- The playable checkpoint is the **75k single-player** model (recognizable, input-responsive Doom
  that drifts on long open-loop rollouts — hence §5).
- **Episodic memory (`MIRA_WORLDKV=1`) is off by default and does not currently help** — it is
  implemented, cheap and correct, but measured no better than baseline on `wm`. See §5d before
  enabling it.

---

## 7. Quick reference — endpoints

| Route | Method | Body / query | Returns |
|---|---|---|---|
| `/` | GET | — | the HTML player |
| `/healthz` | GET | — | `{ready, sessions, video_fps, width, height, checkpoint, ...}` |
| `/ws` | WS | `{keys:[13], turn, reset?}` per message | meta JSON, then consumed-action JSON + JPEG frames pushed as generated |

`keys` is a 13-element 0/1 vector aligned to `DOOM_KEYS`
(`forward, backward, strafe_right, strafe_left, weapon1..7, attack, speed`); `turn` is the mouse
turn-delta, clamped to ±12.
