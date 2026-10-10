"""
client.py, reference client for the udito daemon (and fallback helper).

Two modes:
  python client.py <audio_path> <output_txt> [lang]   # daemon first, one-shot fallback
Used by udito.bat so Hermes gets daemon speed with automatic fallback.
"""
import json
import socket
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORT = 8764


def _cfg() -> dict:
    try:
        from run_transcribe import load_config
        return load_config()
    except Exception:
        return {}

def daemon_transcribe(path: str, lang: str = "") -> dict:
    cfg = _cfg()
    port = int(cfg.get("port", "8764") or 8764)
    attempts = max(1, int(float(cfg.get("retry_attempts", "2") or 2)))
    delay = float(cfg.get("retry_delay", "1.5") or 1.5)
    req = {"cmd": "transcribe", "path": str(path), "lang": lang}
    last_err = None
    for i in range(attempts):
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=2)  # fast fail if daemon down
            s.sendall((json.dumps(req) + "\n").encode())
            s.settimeout(300)
            buf = b""
            while b"\n" not in buf:
                buf += s.recv(65536)
            s.close()
            return json.loads(buf.split(b"\n")[0])
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            last_err = e
            if i < attempts - 1:
                import time
                time.sleep(delay)  # daemon mid-restart: retry before falling back
    raise last_err


def main() -> None:
    if len(sys.argv) < 3:
        print("[udito] usage: client.py <input_audio> <output_txt> [lang]", file=sys.stderr)
        sys.exit(2)
    audio, out, lang = sys.argv[1], sys.argv[2], (sys.argv[3].strip() if len(sys.argv) > 3 else "")
    try:
        r = daemon_transcribe(audio, lang)
        if not r.get("ok"):
            # daemon answered: semantic error. Surface it; never re-pay a 90s
            # one-shot compile just to fail the same way again.
            print(f"[udito] daemon error: {r.get('error')}", file=sys.stderr)
            sys.exit(1)
        Path(out).write_text(r["text"], encoding="utf-8")
        print(f"[udito] daemon: {r['elapsed']}s for {r['audio_dur']}s audio", file=sys.stderr)
        sys.exit(0)
    except (ConnectionRefusedError, socket.timeout, OSError) as e:
        # transport failure only: daemon unreachable/down. One-shot fallback is
        # for THIS case; retries already happened inside daemon_transcribe.
        print(f"[udito] daemon unavailable ({e}); falling back to one-shot (~40-90s compile)", file=sys.stderr)
        cmd = [sys.executable, str(HERE / "run_transcribe.py"), audio, out]
        if lang:
            cmd.append(lang)
        sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()