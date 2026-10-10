"""
Udito one-shot transcription for the Hermes command STT provider.

Usage:
    python run_transcribe.py <input_audio> <output_txt> [language]

Reads device/model/beam settings from config.yaml (same dir). Writes the plain
transcript to <output_txt> and exits 0. Any failure writes the error to stderr
and exits non-zero, fail-visible, never silent.

Designed for the Hermes command-provider shape:
    command: ".../whisper_stt.bat {input_path} {output_path} {language}"
"""
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"[udito] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_config() -> dict:
    """Read config.yaml with a plain parser (no yaml dep needed for our flat key: value shape)."""
    cfg = {}
    cpath = HERE / "config.yaml"
    if not cpath.exists():
        fail(f"config.yaml missing at {cpath}")
    for line in cpath.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        # strip inline comments on unquoted values
        if v and not v.startswith(("'", '"')):
            v = v.split(" #")[0].strip()
        if v and v[0] in "'\"" and v[-1] == v[0] and len(v) >= 2:
            v = v[1:-1].strip()
            # quoted value followed by trailing comment: strip after the closing quote
        elif v and v[0] in "'\"":
            # find the closing quote and cut everything after it (trailing comment case)
            q = v[0]
            end = v.find(q, 1)
            if end > 0:
                v = v[1:end].strip()
        cfg[k] = v
    for required in ("device", "model"):
        if required not in cfg:
            fail(f"config.yaml missing required key: {required}")
    return cfg


def resolve_model_dir(cfg: dict) -> str:
    # model_dir override: local IR folder wins, no download.
    override = (cfg.get("model_dir") or "").strip()
    if override:
        d = Path(override)
        missing = [f for f in ("openvino_encoder_model.xml", "openvino_encoder_model.bin",
                               "openvino_decoder_model.xml", "openvino_decoder_model.bin",
                               "openvino_tokenizer.xml", "openvino_detokenizer.xml")
                   if not (d / f).exists()]
        if missing:
            fail(f"model_dir '{override}' is missing required OpenVINO IR files: {missing}. "
                 "Point model_dir at a full OV export folder (encoder+decoder+tokenizer .xml/.bin).")
        return str(d)
    model = cfg["model"]
    cand = HERE / "models" / model
    if cand.is_dir():
        return str(cand)
    # HF repo id → download/convert via huggingface_hub snapshot (OpenVINO -ov repos are plain files)
    if "/" in model:
        from huggingface_hub import snapshot_download
        print(f"[udito] downloading {model} ... (first use only)", file=sys.stderr)
        path = snapshot_download(repo_id=model, cache_dir=str(HERE / "hf_cache"))
        return path
    fail(f"model '{model}' is neither a folder under models/ nor a HuggingFace repo id")


def main() -> None:
    if len(sys.argv) < 3:
        fail("usage: run_transcribe.py <input_audio> <output_txt> [language] [--device D]")
    audio_path, out_path = sys.argv[1], sys.argv[2]
    if not Path(audio_path).exists():
        fail(f"input audio not found: {audio_path}")
    cfg = load_config()
    # optional --device flag wins over config (benchmark + fallback plumbing)
    if "--device" in sys.argv:
        i = sys.argv.index("--device")
        if i + 1 < len(sys.argv):
            cfg["device"] = sys.argv[i + 1].upper()
    lang_override = sys.argv[3].strip() if len(sys.argv) > 3 and not sys.argv[3].startswith("--") and sys.argv[3].strip() else ""
    language = lang_override or cfg.get("language", "")
    device = cfg["device"].upper()
    beam = int(cfg.get("beam_size", "1") or "1")
    initial_prompt = cfg.get("initial_prompt", "").strip() or None

    t0 = time.perf_counter()
    try:
        import numpy as np
        import openvino as ov
        import openvino_genai as ov_genai
        import librosa
    except ImportError as e:
        fail(f"missing dependency: {e}. Run: pip install -r requirements.txt")

    # Validate the requested device EXISTS before committing, fail-visible, list what's available
    core = ov.Core()
    available = {d.split(".")[0].upper() for d in core.get_available_devices()}
    if device not in available:
        fail(f"device '{device}' not available. Available: {sorted(available)}")

    model_dir = resolve_model_dir(cfg)
    print(f"[udito] model={model_dir} device={device} beam={beam} lang={language or 'auto'}", file=sys.stderr)

    # NOTE: pipeline compile re-pays per process (~60-90s NPU); cache_dir kwarg is UNSUPPORTED
    # in openvino_genai 2026.2 ('cache_dir' NOT_FOUND), that's why the daemon exists.
    compile_t0 = time.perf_counter()
    pipe = ov_genai.WhisperPipeline(str(model_dir), device=device)
    compile_elapsed = time.perf_counter() - compile_t0
    print(f"[udito] pipeline ready in {compile_elapsed:.1f}s", file=sys.stderr)
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(audio) / sr
    # NPU pads to the fixed 30s window. Longer audio gets REFUSED here (fail-visible),
    # never silently trimmed: a "successful" partial transcript is a lie to the caller.
    if device == "NPU" and duration > 30.5:
        fail(f"audio is {duration:.1f}s; the NPU path handles one 30s window per call. "
             "Start the daemon (serve_transcribe.py) and use client.py, which chunks "
             "long audio; or run this one-shot on device GPU/CPU.")
    if device == "NPU":
        import numpy as np
        audio = np.pad(audio, (0, 30 * sr - len(audio)))

    cfg_obj = ov_genai.WhisperGenerationConfig()
    # Load the model's own generation config first (carries lang_to_id etc.), then apply overrides
    gen_cfg_path = os.path.join(model_dir, "generation_config.json")
    if os.path.exists(gen_cfg_path):
        cfg_obj = ov_genai.WhisperGenerationConfig(gen_cfg_path)
    cfg_obj.num_beams = beam
    if language:
        # API expects language in token-bracket form ("<|en|>") per the loaded lang_to_id map
        cfg_obj.language = language if language.startswith("<|") else f"<|{language}|>"
    if initial_prompt:
        cfg_obj.initial_prompt = initial_prompt

    result = pipe.generate(audio, cfg_obj)
    text = result if isinstance(result, str) else (result.texts[0] if getattr(result, "texts", None) else str(result))

    elapsed = time.perf_counter() - t0
    rtf = elapsed / max(duration, 0.1)
    Path(out_path).write_text(text, encoding="utf-8")
    # Telemetry to stderr, stdout stays clean for the transcript file contract
    print(f"[udito] ok: {len(text)} chars, audio {duration:.1f}s, took {elapsed:.1f}s (RTF {rtf:.2f}x)", file=sys.stderr)


if __name__ == "__main__":
    main()