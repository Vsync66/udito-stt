"""Fetch the models udito needs: a Whisper OpenVINO IR export + the emotion ONNX.

Everything downloads from Hugging Face on first use and is resumable. After this
script runs, the daemon works fully offline.

Usage:
    python scripts/fetch_models.py                # default: whisper-small + emotion
    python scripts/fetch_models.py --model base   # whisper-base instead
    python scripts/fetch_models.py --model medium
    python scripts/fetch_models.py --model large-v3-turbo
"""
import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

WHISPER_REPOS = {
    "base": "OpenVINO/whisper-base-int8-ov",
    "small": "OpenVINO/whisper-small-int8-ov",
    "medium": "OpenVINO/whisper-medium-int8-ov",
    "large-v3-turbo": "OpenVINO/whisper-large-v3-turbo-int8-ov",
}
EMOTION_REPO = "onnx-community/wav2vec2-emotion-recognition-ONNX"

REQUIRED_IR = ("openvino_encoder_model.xml", "openvino_encoder_model.bin",
               "openvino_decoder_model.xml", "openvino_decoder_model.bin",
               "openvino_tokenizer.xml", "openvino_detokenizer.xml")


def fetch_whisper(model_key: str) -> str:
    repo = WHISPER_REPOS[model_key]
    print(f"[udito] fetching whisper '{model_key}' from {repo} ...")
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id=repo, cache_dir=str(HERE / "hf_cache"))
    missing = [f for f in REQUIRED_IR if not (Path(path) / f).exists()]
    if missing:
        print(f"[udito] ERROR: downloaded export is missing files: {missing}", file=sys.stderr)
        sys.exit(1)
    print(f"[udito] whisper ok: {path}")
    return path


def fetch_emotion() -> str:
    dest = HERE / "models" / "emotion_onnx"
    marker = dest / "onnx" / "model_q4f16.onnx"
    if marker.exists():
        print(f"[udito] emotion model already present: {dest}")
        return str(dest)
    print(f"[udito] fetching emotion head from {EMOTION_REPO} ...")
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(repo_id=EMOTION_REPO, cache_dir=str(HERE / "hf_cache")))
    dest.mkdir(parents=True, exist_ok=True)
    # lay the files out the way the daemon expects: config.json at the root,
    # the onnx files under onnx/
    if (path / "config.json").exists():
        shutil.copy(path / "config.json", dest / "config.json")
    onnx_dir = dest / "onnx"
    onnx_dir.mkdir(exist_ok=True)
    src = path / "onnx" if (path / "onnx").is_dir() else path
    n = 0
    for f in src.glob("*.onnx"):
        shutil.copy(f, onnx_dir / f.name)
        n += 1
    if not (marker.exists() or list(onnx_dir.glob("model*.onnx"))):
        print("[udito] ERROR: no ONNX model files found in the emotion export", file=sys.stderr)
        sys.exit(1)
    print(f"[udito] emotion ok: {dest} ({n} onnx file(s))")
    return str(dest)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download udito models (resumable, once).")
    ap.add_argument("--model", choices=sorted(WHISPER_REPOS), default="small")
    ap.add_argument("--skip-emotion", action="store_true")
    args = ap.parse_args()
    try:
        from huggingface_hub import snapshot_download  # noqa: F401
    except ImportError:
        print("[udito] missing dependency: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)
    fetch_whisper(args.model)
    if not args.skip_emotion:
        fetch_emotion()
    print("[udito] all models ready; the daemon works offline from here.")


if __name__ == "__main__":
    main()