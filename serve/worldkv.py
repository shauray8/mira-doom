from __future__ import annotations

import os

import numpy as np

DEG_PER_TURN_UNIT = 1.0  # calibrated 1.0075 against ground-truth video
REV = 360.0 / DEG_PER_TURN_UNIT


def _env(name, default, cast=float):
    return cast(os.environ.get(f"MIRA_WORLDKV_{name}", default))


def enabled() -> bool:
    return os.environ.get("MIRA_WORLDKV", "0") not in ("0", "false", "False", "")


class KVBank:
    """Ring of archived per-frame KV keyed by yaw, indexed by `step % capacity`.

    One entry is one latent frame's K/V across the layers that cache (5 of 16 here, since only every
    `time_attention_every`-th block has temporal attention) = 1.97 MB.
    """

    def __init__(self, player, *, slots=None, gb=None, tol_deg=None, spread=None, refresh=None,
                 chunk=None):
        import torch

        self.player = player
        self.W = player.window_size - 1  # KV slots = context latents; z_t holds one more
        self.slots = slots if slots is not None else _env("SLOTS", 2, int)
        self.tol = tol_deg if tol_deg is not None else _env("TOL", 20.0)
        self.spread = spread if spread is not None else _env("SPREAD", 8, int)
        self.refresh = refresh if refresh is not None else _env("REFRESH", 4, int)
        self.chunk = chunk if chunk is not None else _env("CHUNK", 1, int)
        # Only frames already out of working memory are worth retrieving; younger ones are in the
        # window already.
        self.min_age = self.W + 2

        self.layers = [i for i, kv in enumerate(player.kv) if kv is not None]
        self.per_frame_bytes = sum(
            player.kv[i][j][:, :1].numel() * player.kv[i][j].element_size()
            for i in self.layers for j in (0, 1)
        )
        gb = gb if gb is not None else _env("GB", 4.0)
        self.capacity = max(self.W + self.min_age + 8, int(gb * 1e9 / self.per_frame_bytes))

        # Allocated before graph capture and never resized, so archiving can run underneath it.
        self.k_bank, self.v_bank = {}, {}
        for i in self.layers:
            k, v = player.kv[i]
            self.k_bank[i] = torch.empty((self.capacity, k.shape[0], *k.shape[2:]), dtype=k.dtype,
                                         device=k.device)
            self.v_bank[i] = torch.empty((self.capacity, v.shape[0], *v.shape[2:]), dtype=v.dtype,
                                         device=v.device)

        self.step_of = np.full(self.capacity, -1, dtype=np.int64)
        self.yaw_of = np.zeros(self.capacity, dtype=np.float32)
        self.t = -1
        self.yaw = 0.0
        self.armed = False  # setup_graph's warmup steps must not archive
        self._picks: list[int] = []
        self.stats = dict.fromkeys(
            ("steps", "retrieved", "hits", "mean_abs_dyaw", "pinned_steps", "pinned_slots"), 0)

    def clear(self):
        self.step_of[:] = -1
        self.t, self.yaw, self._picks = -1, 0.0, []

    def note_turn(self, turn: float):
        self.yaw = (self.yaw + float(turn) * DEG_PER_TURN_UNIT) % REV

    def _idx_of_step(self, s: int) -> int | None:
        i = s % self.capacity
        return i if s >= 0 and self.step_of[i] == s else None

    def archive(self):
        """The frame just generated is always KV slot -1. ~2 MB of d2d copy."""
        self.t += 1
        i = self.t % self.capacity
        for l in self.layers:
            k, v = self.player.kv[l]
            self.k_bank[l][i].copy_(k[:, -1], non_blocking=True)
            self.v_bank[l][i].copy_(v[:, -1], non_blocking=True)
        self.step_of[i], self.yaw_of[i] = self.t, self.yaw

    def retrieve(self):
        """Refill the reserved slots. Scoring touches only the yaw array; the KV tensors are never
        searched, only moved once chosen."""
        if self.slots <= 0 or self.t < 0:
            return
        if self.stats["steps"] % self.refresh == 0:
            self._picks = self._decide()
        if self._picks:
            self.stats["pinned_steps"] += 1
            self.stats["pinned_slots"] += len(self._picks)
        for slot, entry in self._plan(self._picks):
            for l in self.layers:
                k, v = self.player.kv[l]
                k[:, slot].copy_(self.k_bank[l][entry], non_blocking=True)
                v[:, slot].copy_(self.v_bank[l][entry], non_blocking=True)
        self.stats["steps"] += 1

    def _plan(self, picks: list[int]) -> list[tuple[int, int]]:
        """Retrieved frames at the oldest positions; every remaining slot gets the frame the
        unmodified model would have had there, so "no match" is bit-identical to baseline."""
        out = list(enumerate(picks))
        for s in range(len(picks), self.slots):
            e = self._idx_of_step(self.t - (self.W - 1) + s)
            if e is not None:
                out.append((s, e))
        return out

    def _decide(self) -> list[int]:
        picks: list[int] = []
        eligible = (self.step_of >= 0) & (self.step_of <= self.t - self.min_age)
        if not eligible.any():
            return picks
        cand = np.flatnonzero(eligible)
        d = np.abs((self.yaw_of[cand] - self.yaw + REV / 2) % REV - REV / 2)

        if self.chunk:
            j = int(np.argmin(d))
            if d[j] <= self.tol:
                best = int(self.step_of[cand[j]])
                picks = [e for e in (self._idx_of_step(s)
                                     for s in range(best - self.slots + 1, best + 1)) if e is not None]
                self.stats["mean_abs_dyaw"] += float(d[j])
                self.stats["hits"] += len(picks)
        else:
            for j in np.argsort(d):
                if d[j] > self.tol or len(picks) >= self.slots:
                    break
                s = self.step_of[cand[j]]
                if any(abs(s - self.step_of[p]) < self.spread for p in picks):
                    continue
                picks.append(int(cand[j]))
                self.stats["mean_abs_dyaw"] += float(d[j])
                self.stats["hits"] += 1

        if picks:
            self.stats["retrieved"] += 1
        picks.sort(key=lambda e: self.step_of[e])
        return picks[-self.slots:]

    def summary(self) -> str:
        st, n = self.stats, max(1, self.stats["steps"])
        return (f"bank {self.capacity} frames ({self.capacity * self.per_frame_bytes / 1e9:.2f} GB, "
                f"{self.capacity / 17.5:.0f}s) | pinned on {100 * st['pinned_steps'] / n:.0f}% of steps, "
                f"{st['pinned_slots'] / n:.2f} of {self.slots} slots/step, "
                f"mean |dyaw| {st['mean_abs_dyaw'] / max(1, st['retrieved']):.1f} deg")


def attach(player, **kw) -> KVBank:
    if player.kv is None:  # the cache is built lazily by the first step
        player.step([0] * player.n_keys, 0.0)

    bank = KVBank(player, **kw)
    player.worldkv = bank
    _step, _step_fast, _setup = player.step, player.step_fast, player.setup_graph

    def wrap(fn):
        def inner(key_vec, turn, **kwargs):
            bank.note_turn(turn)
            out = fn(key_vec, turn, **kwargs)
            if bank.armed:
                bank.archive()
                bank.retrieve()
            return out
        return inner

    def setup_graph():
        _setup()
        bank.clear()
        bank.armed = True

    player.step, player.step_fast = wrap(_step), wrap(_step_fast)
    player.setup_graph = setup_graph
    return bank


def install(play_app) -> None:
    """Enable for every player the server creates (called from serve_local)."""
    _make = play_app.GPUWorker._make_player

    def _make_player(self, steps=play_app.N_DIFFUSION_STEPS):
        p = _make(self, steps)
        bank = attach(p)
        if not play_app.USE_CUDA_GRAPH:
            bank.armed = True  # no setup_graph() to arm it
        print(f"[worldkv] {bank.summary()}", flush=True)
        return p

    play_app.GPUWorker._make_player = _make_player
