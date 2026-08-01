"""Serve the Doom world model as a browser-playable game. THE INFERENCE ENTRY POINT.

    ./infer.sh                                  # or: python serve/serve_local.py
    MIRA_STAGE=wm_long MIRA_CTX=78 ./infer.sh   # 2x memory horizon, weaker action control
"""
import asyncio
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

os.environ.setdefault("MIRA_ATTN_BACKEND", "sdpa")

from config import DATA_DIR, ensure_codec_path  # noqa: E402

STAGE = os.environ.get("MIRA_STAGE", "wm")
CTX_FRAMES = int(os.environ.get("MIRA_CTX", "0"))
HOST = os.environ.get("MIRA_HOST", "0.0.0.0")
PORT = int(os.environ.get("MIRA_PORT", "3754"))
COMPILE_DECODER = os.environ.get("MIRA_COMPILE_DECODER", "1") not in ("0", "false", "False")


def _env_flag(name, default=True):
    return os.environ.get(name, "1" if default else "0") not in ("0", "false", "False")


ensure_codec_path()

import play_app  # noqa: E402  -- must follow ensure_codec_path()

play_app.PLAY_STAGE = STAGE
play_app.N_DIFFUSION_STEPS = int(os.environ.get("MIRA_STEPS", "6"))
play_app.NOISE_LEVEL = float(os.environ.get("MIRA_NOISE", "0.0"))
play_app.GUIDANCE = float(os.environ.get("MIRA_GUIDANCE", "1.0"))
play_app.JPEG_QUALITY = int(os.environ.get("MIRA_JPEG_QUALITY", "90"))  # ~9 Mbit/s at 35 fps
play_app.USE_CUDA_GRAPH = _env_flag("MIRA_CUDA_GRAPH", True)
play_app.COMPILE_DECODE = False  # the decoder is compiled explicitly in build_app()

import worldkv  # noqa: E402  -- patches GPUWorker._make_player, so install before the worker starts

if worldkv.enabled():
    worldkv.install(play_app)


_HTML = """<!doctype html><html><head><meta charset=utf-8><title>MIRA — play the world model</title>
<style>
 body{margin:0;background:#08080c;color:#ccd;font:14px/1.5 system-ui;display:flex;
   flex-direction:column;align-items:center;gap:10px;padding:16px}
 canvas{image-rendering:pixelated;width:1024px;max-width:96vw;aspect-ratio:4/3;background:#000;
   border:1px solid #223;cursor:none}
 #hud{display:flex;gap:16px;align-items:center;flex-wrap:wrap;justify-content:center}
 button{background:#1a2a44;color:#cde;border:1px solid #2a4a7a;padding:6px 14px;border-radius:6px;
   cursor:pointer;font:inherit}
 button:hover{background:#24395c}
 kbd{background:#1a1a26;border:1px solid #333;border-radius:4px;padding:1px 6px}
 #stat{font-variant-numeric:tabular-nums;color:#7d9}
 .dim{color:#889;text-align:center;max-width:1024px}
 /* Input display: shows the action the SERVER consumed, i.e. what the model was conditioned on. */
 #inputs{display:flex;gap:22px;align-items:center;background:#0f1118;border:1px solid #1d2030;
   border-radius:10px;padding:10px 18px}
 .grp{display:flex;flex-direction:column;align-items:center;gap:4px}
 .grp .lbl{font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:#667}
 .row{display:flex;gap:4px}
 .k{width:30px;height:30px;display:flex;align-items:center;justify-content:center;border-radius:6px;
   background:#161926;border:1px solid #262b3d;color:#7a8099;font-size:13px;font-weight:600;
   transition:background .05s,color .05s,border-color .05s}
 .k.on{background:#2f7d4f;border-color:#48c07a;color:#eaffe9;box-shadow:0 0 8px #2f7d4f88}
 .k.fire.on{background:#a83a2a;border-color:#e2603f;box-shadow:0 0 8px #a83a2a88}
 .k.wide{width:58px;font-size:11px}
 .k.w{width:24px;height:24px;font-size:11px}
 #turnbar{position:relative;width:150px;height:26px;background:#161926;border:1px solid #262b3d;
   border-radius:6px;overflow:hidden}
 #turnfill{position:absolute;top:0;bottom:0;left:50%;width:0;background:#3d7fd6}
 #turnbar .mid{position:absolute;left:50%;top:0;bottom:0;width:1px;background:#39405a}
 #turnval{font-size:11px;color:#7a8099;font-variant-numeric:tabular-nums}
 input[type=range]{width:130px;accent-color:#3d7fd6}
</style></head><body>
<h2 style="margin:2px">MIRA — a world model you can play</h2>
<canvas id=c width=512 height=384></canvas>
<div id=hud>
  <button id=go>Click to play (locks mouse)</button>
  <button id=reset>New scene (R)</button>
  <label style="font-size:12px;color:#889">sensitivity
    <input type=range id=sens min=5 max=100 value=22> <span id=sensval>0.22</span></label>
  <span id=stat>connecting…</span>
</div>
<div id=inputs>
  <div class=grp><span class=lbl>move</span>
    <div class=row><span class=k id=k_forward>W</span></div>
    <div class=row><span class=k id=k_strafe_left>A</span><span class=k id=k_backward>S</span>
      <span class=k id=k_strafe_right>D</span></div>
  </div>
  <div class=grp><span class=lbl>turn (mouse)</span>
    <div id=turnbar><div class=mid></div><div id=turnfill></div></div>
    <span id=turnval>0.0</span>
  </div>
  <div class=grp><span class=lbl>weapon</span>
    <div class=row id=weprow></div>
  </div>
  <div class=grp><span class=lbl>action</span>
    <div class=row><span class="k wide fire" id=k_attack>FIRE</span>
      <span class="k wide" id=k_speed>RUN</span></div>
  </div>
</div>
<div class=dim><kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> or <kbd>↑</kbd><kbd>←</kbd><kbd>↓</kbd><kbd>→</kbd> move ·
  <b>mouse</b> turn · <b>hold click</b> / <kbd>Space</kbd> fire ·
  <kbd>1</kbd>–<kbd>7</kbd> weapon · <kbd>R</kbd> new scene · <kbd>C</kbd> walk/run ·
  <kbd>Esc</kbd> release mouse<br><span style="font-size:12px">The panel above shows what the
  <b>model</b> received, not what you pressed — weapon and fire are held server-side for a few
  frames, so a tap lights up longer than you held it. Running is on by default. A random weapon is
  selected on every refresh. Press <kbd>R</kbd> whenever the world melts.</span></div>
<script>
const cv=document.getElementById('c'),ctx=cv.getContext('2d',{alpha:false});
const statEl=document.getElementById('stat');
const KEYS=["forward","backward","strafe_right","strafe_left","weapon1","weapon2","weapon3",
  "weapon4","weapon5","weapon6","weapon7","attack","speed"];
const down={}; let turn=0, playing=false, VIDEO_FPS=35;
const km={KeyW:"forward",KeyS:"backward",KeyD:"strafe_right",KeyA:"strafe_left",ShiftLeft:"speed",
  ArrowUp:"forward",ArrowDown:"backward",ArrowRight:"strafe_right",ArrowLeft:"strafe_left"};

// 0.6 saturated the model's +-12-per-step clamp on any real drag: the browser samples at 60Hz but
// the model consumes 17.5 actions/s, so ~3.4 samples sum into each step.
let SENS=0.22;
const sensEl=document.getElementById('sens'),sensValEl=document.getElementById('sensval');
sensEl.oninput=()=>{SENS=sensEl.value/100;sensValEl.textContent=SENS.toFixed(2);
  localStorage.setItem('mira_sens',sensEl.value);};
if(localStorage.getItem('mira_sens')){sensEl.value=localStorage.getItem('mira_sens');sensEl.oninput();}

// Driven by the action the SERVER consumed: weapon and fire are latched server-side in latent
// steps, so only the echo shows what the model was really given.
const weprow=document.getElementById('weprow');
for(let n=1;n<=7;n++){const e=document.createElement('span');e.className='k w';e.id='k_weapon'+n;
  e.textContent=n;weprow.appendChild(e);}
const turnfill=document.getElementById('turnfill'),turnval=document.getElementById('turnval');
function showConsumed(keys,t){
  KEYS.forEach((name,i)=>{const el=document.getElementById('k_'+name);
    if(el) el.classList.toggle('on',!!keys[i]);});
  const frac=Math.max(-1,Math.min(1,t/12));   // the server clamps to +-12 per step
  turnfill.style.width=Math.abs(frac)*50+'%';
  turnfill.style.left=frac>=0?'50%':(50+frac*50)+'%';
  turnval.textContent=t.toFixed(1);
}

// The client reports intent only; holds are latched server-side in latent steps, because the
// browser's send rate is ~3x the rate the model consumes actions.
let weapReq=0, resetReq=false;
addEventListener('keydown',e=>{
  if(km[e.code]) down[km[e.code]]=1;
  if(e.code==='Space'||e.code==='ControlLeft'){down.attack=1;e.preventDefault();}
  if(e.code.startsWith('Digit')){const n=+e.code.slice(5); if(n>=1&&n<=7) weapReq=n;}
  if(e.code==='KeyR') resetReq=true;
  if(e.code==='KeyC') down.walk=!down.walk;   // toggle walk; the model's default state is running
  if(playing) e.preventDefault();
});
document.getElementById('reset').onclick=()=>{resetReq=true;};
addEventListener('keyup',e=>{
  if(km[e.code]) down[km[e.code]]=0;
  if(e.code==='Space'||e.code==='ControlLeft') down.attack=0;
});
document.getElementById('go').onclick=()=>cv.requestPointerLock();
document.addEventListener('pointerlockchange',()=>{playing=(document.pointerLockElement===cv);});
document.addEventListener('mousemove',e=>{if(playing)turn+=e.movementX*SENS;});
// while the pointer is locked, button events go to the document, not the canvas
document.addEventListener('mousedown',()=>{if(playing){down.attack=1;}});
document.addEventListener('mouseup',()=>{down.attack=0;});

function actionVec(){
  const v=KEYS.map(k=>down[k]?1:0);
  // run is on in 88% of the training data, so it is the default state, not a modifier.
  v[KEYS.indexOf("speed")]=down.walk?0:1;
  if(weapReq){v[KEYS.indexOf("weapon"+weapReq)]=1;weapReq=0;}  // one-shot; the server holds it
  const t=turn; turn=0;   // the server accumulates and clamps; send the raw delta
  const r=resetReq; resetReq=false;
  return {keys:v,turn:t,reset:r};
}

// The server pushes frames as generated; they land in a jitter buffer and are drawn on a clock, so
// displayed fps is independent of latency (pull gave a distant client ~7 fps from a server at 21).
let ws=null, buf=[], drawn=0, lastDraw=performance.now(), ema=0, connected=false;
const BUF_TARGET=3;    // frames of cushion before playback starts
const BUF_MAX=10;      // above this we are accumulating lag: drop the oldest to catch up

function connect(){
  const proto=location.protocol==='https:'?'wss':'ws';
  ws=new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType='blob';
  ws.onopen=()=>{connected=true;statEl.textContent='loading model…';};
  ws.onclose=()=>{connected=false;statEl.textContent='disconnected — reconnecting…';
    setTimeout(connect,1000);};
  ws.onmessage=async ev=>{
    if(typeof ev.data==='string'){const m=JSON.parse(ev.data);
      if(m.video_fps){VIDEO_FPS=m.video_fps;cv.width=m.width;cv.height=m.height;}
      if(m.k) showConsumed(m.k,m.t);      // the action the model was conditioned on
      return;}
    const bm=await createImageBitmap(ev.data);
    if(buf.length>BUF_MAX){buf.shift().close();}   // drop oldest, never let lag accumulate
    buf.push(bm);
  };
  // 60Hz, fire-and-forget. The server sums turn deltas, so oversampling loses no mouse motion.
  setInterval(()=>{
    if(ws&&ws.readyState===1) ws.send(JSON.stringify(actionVec()));
  },1000/60);
}

let nextDue=0;
function render(now){
  requestAnimationFrame(render);
  if(buf.length<(nextDue?1:BUF_TARGET)) return;   // wait for cushion, then keep going
  if(!nextDue) nextDue=now;
  if(now<nextDue) return;
  const bm=buf.shift();
  ctx.drawImage(bm,0,0,cv.width,cv.height);
  bm.close();
  const dt=now-lastDraw; lastDraw=now; ema=ema?ema*0.9+dt*0.1:dt; drawn++;
  nextDue+=1000/VIDEO_FPS;
  if(now-nextDue>500) nextDue=now;                // long stall: resync rather than sprint
  if(drawn%10===0) statEl.textContent=
    `${(1000/ema).toFixed(0)} fps · buffer ${buf.length}` + (playing?'':' · click to play');
}
requestAnimationFrame(render);
connect();
</script></body></html>"""


def build_app():
    """Load the model, start the single paced GPU worker, and return the FastAPI app."""
    import time

    import torch
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse

    from mira.data.training_loader import create_loader

    model, loader, ckpt = play_app._load_model_and_seed(STAGE, "doom")

    # Longer memory horizon. n_context_frames is an inference-time rollout knob (weights are
    # unchanged); the ceiling is the trained latent window, video.timesteps.
    if CTX_FRAMES:
        model.set_inference_context(CTX_FRAMES)
        # The loader was built for the previous context length -- rebuild it so the seed clip is
        # long enough to fill the bigger window.
        loader = create_loader(
            index_path=f"{DATA_DIR}/doom_mira/test",
            clip_len=model.config.n_context_frames + 4 * model.temporal_downsampling,
            target_fps=int(model.config.video.fps),
            batch_size=1,
            num_workers=2,
            frame_size=(384, 512),
            valid_keys=list(play_app.DOOM_KEYS),
            seed=int(time.time()) % 100000,
            infinite=True,
        )

    if COMPILE_DECODER:
        # -no-cudagraphs: InteractivePlayer.setup_graph captures its own graph around the whole
        # step, and inductor's cudagraph wrapper would fight it.
        model.codec.decoder = torch.compile(
            model.codec.decoder, mode="max-autotune-no-cudagraphs", dynamic=False
        )

    td = model.temporal_downsampling
    video_fps = float(model.config.video.fps)
    realtime_hz = video_fps / td  # latent steps per second of real time
    pace_hz = float(os.environ.get("MIRA_PACE", realtime_hz))

    horizon_s = model.n_context_latents * td / video_fps
    print(f"[serve] checkpoint    {ckpt}")
    print(f"[serve] context       {model.n_context_frames} frames "
          f"({model.n_context_latents} latents) = {horizon_s:.2f}s of memory")
    print(f"[serve] diffusion     {play_app.N_DIFFUSION_STEPS} steps, noise {play_app.NOISE_LEVEL}")
    print(f"[serve] pacing        {pace_hz:.1f} latent steps/s -> {pace_hz * td:.0f} video fps "
          f"(real-time = {realtime_hz:.1f})")
    print(f"[serve] compiled decoder={COMPILE_DECODER} cuda_graph={play_app.USE_CUDA_GRAPH}")

    worker = play_app.GPUWorker(model, loader, pace_hz=pace_hz or None)
    worker.start()
    if not worker.ready.wait(timeout=1800):
        raise RuntimeError("GPU worker did not become ready within 1800s")
    print("[serve] ready", flush=True)

    api = FastAPI()
    meta = {
        "video_fps": video_fps,
        "width": model.config.video.width,
        "height": model.config.video.height,
        "checkpoint": ckpt,
        "context_seconds": round(horizon_s, 2),
        "diffusion_steps": play_app.N_DIFFUSION_STEPS,
    }

    @api.get("/")
    def index():
        return HTMLResponse(_HTML)

    @api.get("/healthz")
    def healthz():
        return {"ready": worker.ready.is_set(), "sessions": len(worker.sessions), **meta}

    @api.websocket("/ws")
    async def ws_play(ws: WebSocket):
        await ws.accept()
        sid = worker.request_session(play_app.N_DIFFUSION_STEPS)
        for _ in range(1200):  # the worker thread attaches the session on its next loop pass
            if sid in worker.sessions:
                break
            await asyncio.sleep(0.05)
        sess = worker.sessions.get(sid)
        if sess is None:
            await ws.close(code=1011)
            return
        await ws.send_text(json.dumps(meta))
        print(f"[serve] ws session {sid}", flush=True)

        async def pump():
            """Awaiting send_bytes applies backpressure; the session's bounded queue drops what
            piles up, so a slow client falls behind in quality, never in time. Each batch is preceded
            by the action the model was CONDITIONED on -- not what the browser sent."""
            last_sent = None
            while True:
                frames = sess.drain()
                if not frames:
                    await asyncio.sleep(0.004)
                    continue
                keys, turn = sess.last_consumed
                state = {"k": keys, "t": round(turn, 2)}
                if state != last_sent:
                    await ws.send_text(json.dumps(state))
                    last_sent = state
                for _fid, jpeg in frames:
                    await ws.send_bytes(jpeg)

        pump_task = asyncio.create_task(pump())
        try:
            while True:
                body = json.loads(await ws.receive_text())
                sess.set_action(body["keys"], float(body["turn"]), reset=body.get("reset", False))
        except (WebSocketDisconnect, RuntimeError, json.JSONDecodeError):
            pass
        finally:
            pump_task.cancel()
            worker.sessions.pop(sid, None)
            print(f"[serve] ws closed {sid}", flush=True)

    return api


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(build_app(), host=HOST, port=PORT, log_level="warning", ws_max_size=4 * 1024 * 1024)
