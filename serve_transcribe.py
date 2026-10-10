"""
udito daemon, resident transcription service (NPU/GPU/CPU).

Loads the OpenVINO Whisper pipeline ONCE (~60-90s NPU compile), then serves
sub-second transcription over a localhost TCP port. Hermes command providers
(or anything) POST the audio path; the daemon returns the transcript.

Wire protocol (line-delimited JSON over TCP):
    REQ {"cmd":"transcribe","path":"<wav/mp3 path>","lang":"en"}        \\n
    RESP {"ok":true,"text":"...","elapsed":0.7,"audio_dur":7.4}          \\n
    REQ {"cmd":"ping"}                                                   \\n
    RESP {"ok":true,"text":"pong"}
    REQ {"cmd":"shutdown"}
    RESP {"ok":true}

Run:  python serve_transcribe.py           (uses config.yaml, port 8764)
      python serve_transcribe.py --port 8764 --device NPU
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_transcribe import load_config, resolve_model_dir, fail  # reuse the shared config


def get_emotion_session(cfg=None):
    """wav2vec2 emotion head on OpenVINO CPU (one runtime for the whole
    product; measured live: +51MB RAM vs onnxruntime's +574MB session
    arena, faster first call, same outputs). Emotion IR lives at
    models/<sel>/ov/emotion.xml. .reset() drops the session so the
    unload command frees the RAM."""
    if cfg == "reset":
        get_emotion_session._sess = None
        get_emotion_session._key = None
        global _EMOTION_LABELS
        _EMOTION_LABELS = None
        return None
    cfg = cfg or {}
    model_sel = (cfg.get("emotion_model") or "emotion_onnx").strip()
    key = model_sel
    if not hasattr(get_emotion_session, "_sess") or getattr(get_emotion_session, "_key", None) != key:
        import openvino as ov
        mdir = Path("models") / model_sel
        ir = mdir / "ov" / "emotion.xml"
        if not ir.exists():
            # one-time conversion: ONNX export -> OpenVINO IR (writes ov/)
            onnx_path = mdir / "onnx" / "model_q4f16.onnx"
            if not (onnx_path.exists() or (mdir / "onnx").is_dir()):
                raise FileNotFoundError(f"emotion model '{model_sel}' not found under models/. "
                                         f"Run 'python main.py models' to fetch it, or set emotion_model to an "
                                         f"existing folder under models/.")
            src_onnx = onnx_path if onnx_path.exists() else next((mdir / "onnx").glob("model*.onnx"), None)
            if src_onnx is None:
                raise FileNotFoundError(f"no ONNX export found in {mdir}/onnx/ for conversion")
            ir.parent.mkdir(parents=True, exist_ok=True)
            core = ov.Core()
            ov.save_model(core.read_model(str(src_onnx)), str(ir))
            print(f"[udito] emotion IR converted: {ir}", file=sys.stderr)
        core = ov.Core()
        get_emotion_session._sess = core.compile_model(str(ir), "CPU")
        get_emotion_session._key = key
        _EMOTION_LABELS = None  # reload labels from the selected model's config
    return get_emotion_session._sess


_EMOTION_LABELS = None

def classify_emotion(path: str, cfg: dict = None, audio_arr=None) -> dict:
    """Return top emotion + probabilities for the audio at path (16k mono prep).
    Uses the selected emotion model via get_emotion_session.
    audio_arr: pass the already-decoded 16k mono array when the caller just
    transcribed the same file, avoiding a second disk read."""
    global _EMOTION_LABELS
    import json as _json
    import numpy as np
    import librosa
    cfg = cfg or {}
    model_sel = (cfg.get("emotion_model") or "emotion_onnx").strip()
    sess = get_emotion_session(cfg)
    if _EMOTION_LABELS is None:
        cfgj = _json.load(open(HERE / "models" / model_sel / "config.json"))
        _EMOTION_LABELS = cfgj["id2label"]
    if audio_arr is not None:
        audio = np.asarray(audio_arr, dtype=np.float32)
    else:
        audio, _sr = librosa.load(path, sr=16000, mono=True)
    if len(audio) > 16000 * 30:  # head matches whisper's window
        audio = audio[: 16000 * 30]
    # wav2vec2 is trained on zero-mean/unit-variance audio (fairseq convention).
    # Feeding raw RMS-level float32 made every take read as dominant negative
    # valence ("disgust 0.93" on all quantizations); the raw-vs-normalized
    # comparison proved the prep step was the cause, not quantization.
    audio = (audio - audio.mean()) / (audio.std() + 1e-7)
    logits = sess(np.asarray(audio, dtype=np.float32)[None, :])[sess.output(0)][0]
    e = np.exp(logits - logits.max()); p = e / e.sum()
    ranked = sorted(((_EMOTION_LABELS[str(i)], float(p[i])) for i in range(len(p))),
                    key=lambda x: -x[1])
    return {"ok": True, "emotion": ranked[0][0], "confidence": round(ranked[0][1], 3),
            "all": {k: round(v, 4) for k, v in ranked}}


def load_pipeline(cfg, device=None):
    import openvino as ov
    import openvino_genai as ov_genai

    # device: auto probes the machine, best available wins,
    # choice reported. NPU > GPU > CPU. Unknown device names still fail loudly.
    device = (device or cfg["device"]).strip().upper()
    core = ov.Core()
    available = {d.split(".")[0].upper() for d in core.get_available_devices()}
    if device == "AUTO":
        for pref in ("NPU", "GPU", "CPU"):
            if pref in available:
                device = pref
                break
        else:
            fail(f"device 'auto': no OpenVINO device found. Available: {sorted(available)}")
        print(f"[udito] device auto -> {device} (available: {sorted(available)})", file=sys.stderr)
    elif device not in available:
        fail(f"device '{device}' not available. Available: {sorted(available)}, "
             f"use device: auto to probe, or pick from the list.")

    model_dir = resolve_model_dir(cfg)
    print(f"[udito] loading model={Path(model_dir).name} device={device} ...", file=sys.stderr)
    t0 = time.perf_counter()
    # compile cache (measured 2.5s vs 39-70s live): OpenVINO
    # saves the compiled NPU blob to CACHE_DIR; later boots import instead of
    # recompiling. SELF-HEALING: blobs are driver-specific, after a driver
    # update the old blob fails to import, so we wipe + recompile + re-cache.
    compile_cache = str(cfg.get("compile_cache", "true")).strip().lower() in ("true", "1", "yes")
    if not compile_cache:
        pipe = ov_genai.WhisperPipeline(str(model_dir), device=device)
        print(f"[udito] pipeline ready in {time.perf_counter()-t0:.1f}s (cache off)", file=sys.stderr)
        return pipe, device

    cache_root = cfg.get("compile_cache_dir", "") or str(HERE / "blob_cache")
    cache_dir = os.path.join(cache_root, device.lower())
    os.makedirs(cache_dir, exist_ok=True)

    def _try_load(cache_prop: bool):
        kw = {"CACHE_DIR": cache_dir} if cache_prop else {}
        return ov_genai.WhisperPipeline(str(model_dir), device=device, **kw)

    t_try = time.perf_counter()
    try:
        pipe = _try_load(True)
        print(f"[udito] pipeline ready in {time.perf_counter()-t_try:.1f}s "
              f"(cache: {'hit' if (time.perf_counter()-t_try) < 20 else 'cold-write'}, {cache_dir})", file=sys.stderr)
    except Exception as e:
        # stale/foreign blob (driver update, device change): wipe, recompile cold, re-cache
        import shutil
        print(f"[udito] cache import failed ({type(e).__name__}), wiping {cache_dir} "
              f"and recompiling (driver updated?) ...", file=sys.stderr)
        shutil.rmtree(cache_dir, ignore_errors=True)
        os.makedirs(cache_dir, exist_ok=True)
        t_cold = time.perf_counter()
        pipe = _try_load(True)
        print(f"[udito] pipeline ready in {time.perf_counter()-t_cold:.1f}s "
              f"(cache rebuilt after wipe)", file=sys.stderr)
    return pipe, device


def _denoise_ffmpeg(path: str, cfg: dict) -> str:
    """Noise reduction stage: off|afftdn|rnnoise, applied as a
    filter during the 16k decode pass, one ffmpeg invocation, no extra file hops.
    Returns the same path when off."""
    mode = str(cfg.get("noise_reduction", "off") or "off").strip().lower()
    if mode == "off":
        return path
    if mode == "afftdn":
        af = "afftdn=nr=12:nf=-25"
    elif mode == "rnnoise":
        m = cfg.get("noise_reduction_model", "models/rnnoise/std.rnnn")
        mpath = Path(m if os.path.isabs(str(m)) else str(HERE / m))
        if not mpath.exists():
            raise FileNotFoundError(f"rnnoise model not found: {mpath} (config noise_reduction_model)")
        # Windows drive-colons in arnndn model paths do not parse in any escape
        # scheme this ffmpeg build accepts, but cwd-relative filenames parse fine.
        # So the ffmpeg run happens with cwd=model dir + bare filename. No shell.
        af = "arnndn=m=" + mpath.name
        cwd = str(mpath.parent)
    else:
        raise ValueError(f"unknown noise_reduction mode: {mode!r} (off|afftdn|rnnoise)")
    out = os.path.join(tempfile.gettempdir(),
                       f"udito_denoise_{os.getpid()}_{uuid.uuid4().hex[:8]}.wav")
    r = subprocess.run([_FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                        "-i", path, "-af", af, "-ar", "16000", "-ac", "1", out],
                       capture_output=True, text=True,
                       cwd=cwd if mode == "rnnoise" else None)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg denoise failed: {r.stderr[-300:]}")
    return out


_FFMPEG = shutil.which("ffmpeg")


def _decode_to_wav16k(path: str) -> str:
    """Any input format (webm/opus/mp3/m4a/wav) -> 16kHz mono wav via ffmpeg.
    Temp output gets a uuid suffix so parallel requests never collide."""
    if not _FFMPEG:
        raise RuntimeError("ffmpeg not found on PATH. Install ffmpeg and retry.")
    out = os.path.join(tempfile.gettempdir(),
                       f"udito_decode_{os.getpid()}_{uuid.uuid4().hex[:8]}.wav")
    r = subprocess.run([_FFMPEG, "-y", "-i", path, "-ar", "16000", "-ac", "1", out],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0 or not Path(out).exists():
        raise RuntimeError(f"ffmpeg decode failed: {r.stderr[-300:]}")
    return out


def _boost_if_quiet(audio, sr: int, cfg: dict):
    """RMS-based gain for quiet takes. Quiet audio makes Whisper return empty/hallucinated
    transcripts (no_speech_prob fires on low energy). Boost = scale so RMS reaches target,
    clamped so peaks never clip. Config: boost_enabled (default true), boost_target_rms
    (default 0.05, roughly normal speaking level on mic input), boost_max_gain (default 8x)."""
    import numpy as np
    if str(cfg.get("boost_enabled", "true")).strip().lower() not in ("true", "1", "yes"):
        return audio, None
    target = float(cfg.get("boost_target_rms", "0.05") or 0.05)
    max_gain = float(cfg.get("boost_max_gain", "8") or 8)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms <= 1e-6:  # silence: nothing to boost, don't invent noise
        return audio, {"rms": 0.0, "gain": 0.0}
    gain = min(target / rms, max_gain)
    if gain <= 1.02:  # already loud enough
        return audio, {"rms": round(rms, 4), "gain": 1.0}
    boosted = audio * gain
    peak = float(np.max(np.abs(boosted)))
    if peak > 0.98:  # clamp to avoid clipping distortion
        boosted = boosted * (0.98 / peak)
    return boosted, {"rms": round(rms, 4), "gain": round(gain, 2)}


def _decorate_pauses(text: str, audio, sr: int, pause_thr: float = 0.6) -> tuple:
    """One symbol, real pauses. The VAD finds acoustic silences; each silence
    over pause_thr seconds becomes '...' between the last word before it and
    the first word after it, via the uniform-rate word mapping WITHIN each
    speech run (small span = small error; stretching the mapping across the
    whole take instead overmarked).
    No periods, no capitals, no grammar model. Returns (decorated_text, hints)."""
    import numpy as np
    words = text.split()
    runs, frame = _speech_energy_runs(audio, sr)
    if not words or len(runs) < 2:
        return text, {"gaps": 0}
    total_speech = sum(b - a for a, b in runs)
    marks = {}
    cum = 0
    for k in range(len(runs) - 1):
        cum += runs[k][1] - runs[k][0]
        gap_s = (runs[k + 1][0] - runs[k][1]) / sr
        if gap_s < pause_thr:
            continue
        boundary = round(cum / total_speech * len(words))
        if 0 < boundary < len(words) and boundary not in marks:
            marks[boundary] = gap_s  # longest gap at a shared boundary wins
    if not marks:
        return text, {"gaps": 0}
    out = []
    for wi, w in enumerate(words):
        out.append(w)
        if (wi + 1) in marks:
            out.append("...")
    return " ".join(out), {"gaps": len(marks), "mode": "ellipsis-vad"}


def _speech_energy_runs(audio, sr: int):
    """Shared VAD front-end: frame-wise RMS -> speech runs [(start,end) frames].
    Noise floor = 10th percentile, not the median (the median IS speech on
    speech-dense audio, which swallows the gaps)."""
    import numpy as np
    frame = int(sr * 0.02)
    n = len(audio) // frame
    if n == 0:
        return [], frame
    e = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1))
    noise = float(np.percentile(e, 10)) + 1e-6
    thresh = max(noise * 3.0, float(np.percentile(e, 90)) * 0.25)
    speech = e > thresh
    runs = []
    start = None
    for i, s in enumerate(speech):
        if s and start is None:
            start = i
        elif not s and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, n))
    return [(a * frame, b * frame) for a, b in runs if b - a >= 10], frame


def transcribe(pipe, cfg, path: str, lang_override: str = "", punct_mode: str = "off") -> dict:
    import numpy as np
    import librosa

    t0 = time.perf_counter()
    audio, sr = librosa.load(path, sr=16000, mono=True)
    duration = len(audio) / sr
    # Silence gate: no speech energy above the VAD floor means
    # the only thing Whisper would produce here is a hallucination ("Thank you.").
    import numpy as _np
    _rms = float(_np.sqrt(_np.mean(audio ** 2))) if len(audio) else 0.0
    if _rms < 1e-4:
        return {"ok": True, "text": "", "elapsed": round(time.perf_counter() - t0, 2),
                "audio_dur": round(duration, 2), "speech_detected": False,
                "boost": {"rms_global": round(_rms, 6), "mode": "per-segment"}}
    # production guard : refuse pathologically long takes fast
    max_audio = float(cfg.get("max_audio_seconds", "300") or 300)
    if duration > max_audio:
        return {"ok": False, "error": f"audio too long: {duration:.0f}s > max_audio_seconds "
                                      f"{max_audio:.0f}. Split the file or raise the limit in config."}

    import openvino_genai as ov_genai
    cfg_obj = ov_genai.WhisperGenerationConfig()
    gen_cfg_path = os.path.join(resolve_model_dir(cfg), "generation_config.json")
    if os.path.exists(gen_cfg_path):
        cfg_obj = ov_genai.WhisperGenerationConfig(gen_cfg_path)
    beam = int(cfg.get("beam_size", "1") or "1")
    cfg_obj.num_beams = beam
    language = (lang_override or cfg.get("language", "")).strip()
    if language:
        cfg_obj.language = language if language.startswith("<|") else f"<|{language}|>"
    initial_prompt = (cfg.get("initial_prompt", "") or "").strip()
    if initial_prompt:
        cfg_obj.initial_prompt = initial_prompt  # only reaches here if boot validation passed (NPU+prompt=ROI bug, checked at startup)

    # quiet-take boost, PER-SEGMENT, not take-global: a quiet segment inside an
    # overall-loud take got gain 1.0 under take-global RMS and Whisper dropped it
    # entirely. Each segment is measured alone; the take-global RMS is kept in
    # the response for continuity.
    audio_rms_global = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0

    # one symbol, real pauses: the VAD measures acoustic silences; anything
    # over pause_ms becomes "..." between the words it separates.
    pause_thr = int(cfg.get("pause_ms", "600") or 600) / 1000.0

    def _gen(seg: np.ndarray) -> str:
        seg, _b = _boost_if_quiet(seg, sr, cfg)  # local measurement, local gain
        WINDOW = 30 * sr
        if len(seg) < WINDOW:
            seg = np.pad(seg, (0, WINDOW - len(seg)))
        result = pipe.generate(seg, cfg_obj)
        return result.texts[0] if hasattr(result, "texts") and result.texts else str(result)

    # single coherent transcription pass (<=30s one window; >30s sliding stitch),
    # then decorate ONCE with the VAD pause map
    if duration <= 30.0:
        text = _gen(audio)
    else:
        WINDOW = 30 * sr
        OVERLAP = sr
        step = WINDOW - OVERLAP
        pieces = []
        i = 0
        while i < len(audio):
            seg = audio[i : i + WINDOW]
            if len(seg) < sr * 2 and i > 0:
                break
            pieces.append(_gen(seg))
            i += step
        # dedupe the overlap: words in the final second of each window were
        # transcribed by the next window too. Drop the tail words of each piece
        # that repeat in the head of the following piece.
        for k in range(len(pieces) - 1):
            tail = pieces[k].split()
            head = pieces[k + 1].split()
            # find the longest suffix of `tail` that prefixes `head` (max = overlap/0.3s of speech)
            for n in range(min(len(tail), len(head), 12), 0, -1):
                if tail[-n:] == head[:n]:
                    pieces[k] = " ".join(tail[:-n])
                    break
        text = " ".join(p for p in pieces if p).strip()

    if punct_mode == "pause":
        text, hints = _decorate_pauses(text, audio, sr, pause_thr)
        return {"ok": True, "text": text, "elapsed": round(time.perf_counter() - t0, 2),
                "audio_dur": round(duration, 2), "punct": "pause", "hints": hints,
                "_audio16k": audio,
                "boost": {"rms_global": round(audio_rms_global, 4), "mode": "per-segment"}}
    elapsed = time.perf_counter() - t0
    return {"ok": True, "text": text, "elapsed": round(elapsed, 2), "audio_dur": round(duration, 2),
            "_audio16k": audio,
            "boost": {"rms_global": round(audio_rms_global, 4), "mode": "per-segment"}}


def _singleton_guard(port: int) -> None:
    """One daemon per machine. If something answers ping on the
    port, report it and exit clean, never double-compile (two resident graphs =
    2x RAM + racing requests + a wasted 3.5min compile)."""
    import socket as _s
    try:
        c = _s.create_connection(("127.0.0.1", port), timeout=2)
        c.sendall(b'{"cmd":"ping"}\n')
        buf = b""
        while b"\n" not in buf:
            buf += c.recv(4096)
        c.close()
        r = json.loads(buf.split(b"\n")[0].decode())
        if r.get("ok") and r.get("text") == "pong":
            print(f"[udito] FATAL: daemon already running on :{port}, exit, don't double-compile.", file=sys.stderr)
            sys.exit(2)
    except (ConnectionRefusedError, TimeoutError, socket.timeout, OSError):
        pass  # port free, we're the one


def _validate_boot_config(cfg: dict, resolved_device: str = None) -> None:
    """Fail at boot, not at request. A config that asks this
    daemon for something it can't deliver must stop startup loudly with the reason.
    Intel-exclusive truths live here (NPU ROI prompt bug, beam reshape bug).
    resolved_device: the device load_pipeline actually picked (auto resolves there),
    because laws must apply to the hardware, not the config string."""
    device = (resolved_device or cfg.get("device") or "").strip().upper()
    prompt = (cfg.get("initial_prompt") or "").strip()
    if device == "NPU" and prompt:
        fail("initial_prompt is set but the resolved device is NPU; Intel genai ROI bug "
             "crashes the pipeline on prompt+start_token. Fix: set device: GPU/CPU for "
             "prompt conditioning, or clear initial_prompt.")
    beam = int(cfg.get("beam_size", "1") or "1")
    if device == "NPU" and beam > 1:
        fail(f"beam_size={beam} but device=NPU, Intel remote-tensor reshape bug "
             f"(openvino 2026.2). beam 1 is the stable NPU path; GPU/CPU can use 3-5.")
    if str(cfg.get("concurrency", "sequential")).lower() not in ("sequential", "parallel"):
        fail("concurrency must be 'sequential' or 'parallel'")
    if str(cfg.get("concurrency", "sequential")).lower() == "parallel" and device == "NPU":
        print("[udito] WARNING: parallel on NPU is dubious (one tensor "
              "processor); keeping sequential semantics for NPU regardless.", file=sys.stderr)
    # NPU int8 law: the NPU compiles int8 IR only, raise at boot if the
    # resolved model can't be verified as an int8 OV export (name-in-dir or repo id says -int8).
    if device == "NPU":
        resolved = resolve_model_dir(cfg)
        rid = "int8" in str(cfg.get("model", "")).lower() or "int8" in Path(resolved).name.lower()
        xml = next(Path(resolved).glob("openvino_encoder_model.xml"), None)
        if not (rid or (xml and "int8" in xml.read_text(encoding="utf-8", errors="ignore")[:2000].lower())):
            fail(f"device=NPU but model '{cfg.get('model')}' can't be verified int8, the NPU only "
                 "accepts int8-compiled IR. Fix: pick an *-int8-ov model, set model_dir to an int8 "
                 "IR folder, or switch device to GPU/CPU for fp16/fp32 models.")
    # Emotion selection: model must resolve; IR auto-converts from the
    # ONNX export on first use.
    em = (cfg.get("emotion_model") or "").strip()
    if em:
        em_dir = Path("models") / em
        if not em_dir.is_dir():
            avail = [p.name for p in Path("models").iterdir() if p.is_dir()] if Path("models").is_dir() else []
            fail(f"emotion_model '{em}' not found under models/. Available: {avail}. "
                 "(Or pass an HF repo id for an ONNX-exported wav2vec2-class model.)")
        if not (em_dir / "config.json").exists():
            fail(f"emotion_model '{em}' has no config.json (label map missing).")
    # Pause dial sanity: "..." threshold in ms.
    try:
        pm = int(cfg.get("pause_ms", "600") or 600)
        if not (100 <= pm <= 5000):
            fail(f"pause_ms={pm} outside sane range 100-5000")
    except (TypeError, ValueError):
        fail(f"pause_ms must be an integer; got '{cfg.get('pause_ms')}'")


def serve(port: int = None) -> None:
    _singleton_guard(port if port else int(load_config().get("port", "8764") or 8764))
    cfg = load_config()
    if port is None:
        port = int(cfg.get("port", "8764") or 8764)  # config is the default; CLI flag overrides
    # bind BEFORE the compile: a port collision must fail in
    # one second, never after 40s of NPU compile.
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind(("127.0.0.1", port))
    except OSError as e:
        fail(f"port {port} is already in use ({e}). Another daemon may be running; "
             "check 'netstat -ano | findstr 8764' or change port in config.yaml.")
    srv.listen(8)
    pipe_box = [None]  # mutable box: the unload command drops the model from RAM
    pipe_box[0], device = load_pipeline(cfg)
    _validate_boot_config(cfg, resolved_device=device)

    def _get_pipe():
        # rebuild path after unload: blob cache makes this ~2s, not 40s
        if pipe_box[0] is None:
            print("[udito] reloading pipeline from blob cache...", file=sys.stderr)
            pipe_box[0], _d = load_pipeline(cfg)
        return pipe_box[0]
    lock = threading.Lock()  # one generation at a time (law: sequential default)
    parallel = (str(cfg.get("concurrency", "sequential")).lower() == "parallel"
                and device != "NPU")
    if parallel:
        print("[udito] concurrency=parallel (GPU/CPU): requests run concurrently.", file=sys.stderr)
    global _T0
    _T0 = time.perf_counter()
    print(f"[udito] serving on 127.0.0.1:{port} (device={device})", file=sys.stderr)
    warmup(cfg)

    def handle(conn: socket.socket) -> None:
        try:
            conn.settimeout(300)
            buf = b""
            while b"\n" not in buf:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                buf += chunk
            req = json.loads(buf.split(b"\n")[0].decode("utf-8"))
            cmd = req.get("cmd", "")
            if cmd == "ping":
                resp = {"ok": True, "text": "pong"}
            elif cmd == "health":
                # production monitoring : what's resident, config shape, uptime
                resp = {"ok": True, "device": device, "model": str(cfg.get("model") or Path(resolve_model_dir(cfg)).name).split("/")[-1],
                        "uptime_s": round(time.perf_counter() - _T0, 1),
                        "punct_default": (cfg.get("punct_default", "off") or "off"),
                        "concurrency": ("parallel" if parallel else "sequential"),
                        "max_audio_s": int(float(cfg.get("max_audio_seconds", "300") or 300)),
                        "emotion_model": "loaded" if getattr(get_emotion_session, "_sess", None) else "not-loaded"}
            elif cmd == "shutdown":
                conn.sendall((json.dumps({"ok": True, "text": "bye"}) + "\n").encode())
                conn.close()
                os._exit(0)
            elif cmd == "unload":
                # RAM dial: drop whisper + emotion from RAM
                # without killing the daemon. Next transcribe rebuilds the pipe
                # (~2s from blob cache); next emotion request reloads the IR.
                # The socket stays up; callers probe readiness with ping.
                pipe_box[0] = None
                get_emotion_session("reset")
                resp = {"ok": True, "text": "unloaded (whisper + emotion dropped)"}
            elif cmd == "transcribe":
                path = req.get("path", "")
                if not path or not Path(path).exists():
                    resp = {"ok": False, "error": f"audio not found: {path}"}
                else:
                    # punct surface: off|pause only.
                    # off = raw. pause = "..." decoration from VAD silences
                    # (threshold = pause_ms). Neural punctuation was removed:
                    # "..." never wrongly claims a sentence boundary.
                    punct = str(req.get("punct", "") or "").strip().lower()
                    if not punct:
                        punct = str(cfg.get("punct_default", "pause") or "pause").strip().lower()
                    if punct not in ("off", "pause"):
                        resp = {"ok": False, "error": f"unknown punct mode: {punct!r} "
                                                       "(off|pause; pause does \"...\" from real pauses)"}
                    else:
                        # concurrency is a config choice.
                        # sequential (default, and always for NPU): lock held per request.
                        # parallel (GPU/CPU only): lock not taken, requests run concurrently.
                        # bind the override copy only when there IS an override, so
                        # req_cfg either is cfg or a real copy
                        req_cfg = cfg
                        noise_sel = str(req.get("noise_reduction", "") or "").strip().lower()
                        if noise_sel:
                            req_cfg = dict(cfg)
                            req_cfg["noise_reduction"] = noise_sel
                            print(f"[udito] request noise_reduction override: {noise_sel}", file=sys.stderr)
                        boost_sel = req.get("boost")
                        if boost_sel is not None:  # per-request override (A/B benchmarks, raw mode)
                            if req_cfg is cfg:
                                req_cfg = dict(cfg)
                            req_cfg["boost_enabled"] = "false" if not boost_sel else "true"
                        if parallel:
                            path2 = _decode_to_wav16k(path)
                            path2 = _denoise_ffmpeg(path2, req_cfg)
                            resp = transcribe(_get_pipe(), req_cfg, path2, req.get("lang", "") or "",
                                              punct_mode=punct)
                            resp["pipe"] = device
                        else:
                            with lock:
                                path2 = _decode_to_wav16k(path)
                                path2 = _denoise_ffmpeg(path2, req_cfg)
                                resp = transcribe(_get_pipe(), req_cfg, path2, req.get("lang", "") or "",
                                                  punct_mode=punct)
                                resp["pipe"] = device
                        # Emotion tier: optional annotation; the tag rides in
                        # the transcript itself, not a side field. Skipped on
                        # silence (the head would just read an artifact off
                        # zeros, and the empty transcript stays clean).
                        if resp.get("ok") and bool(req.get("emotion", False)) \
                                and resp.get("speech_detected") is not False:
                            t1 = time.perf_counter()
                            emo = classify_emotion(path2, cfg=req_cfg,
                                                   audio_arr=resp.pop("_audio16k", None))
                            resp["emotion"] = emo
                            resp["emotion_elapsed"] = round(time.perf_counter() - t1, 3)
                            if emo.get("ok") and str(req_cfg.get("emotion_in_text", "true")).lower() != "false":
                                resp["text"] = (resp.get("text") or "").rstrip() + \
                                    f" [{emo.get('emotion', '?')} {emo.get('confidence', 0):.2f}]"
                        else:
                            resp.pop("_audio16k", None)  # never leak arrays to the wire
                    print(f"[udito] {Path(path).name} → {resp.get('elapsed', '?')}s "
                          f"[{resp.get('pipe', '?')}] punct={punct}", file=sys.stderr)
            else:
                resp = {"ok": False, "error": f"unknown cmd: {cmd}"}
            conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
        except Exception as e:  # noqa: BLE001, daemon must survive bad clients
            try:
                conn.sendall((json.dumps({"ok": False, "error": str(e)}) + "\n").encode("utf-8"))
            except Exception:
                pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    while True:
        conn, _ = srv.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


def warmup(cfg=None):
    """Boot warmup: pay the lazy-load tax at boot, never on a voice note.
    Emotion weights stay OUT of RAM until a caller actually asks for emotion.
    An idle daemon holds whisper only; the first emotion request pays ~0.3s
    once, then it stays resident."""
    print("[udito] warmup: whisper resident, emotion loads on first request", file=sys.stderr)


if __name__ == "__main__":
    # config.yaml's port is the default; --port N overrides it (used by tools
    # that need a spare port, e.g. parallel test daemons)
    args = sys.argv[1:]
    if "--port" in args:
        port = int(args[args.index("--port") + 1])
        serve(port)
    else:
        serve()