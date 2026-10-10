"""Udito control center: one entry point for everything.

    python main.py start       # boot headless (log -> daemon.log), wait for ready
    python main.py window      # boot in THIS console: live log, Ctrl+C stops it
    python main.py stop        # polite shutdown over the wire
    python main.py restart     # stop, wait for the port to free, start
    python main.py status      # up/down, device, model, uptime, emotion state
    python main.py log [n]     # last n lines of daemon.log (default 40), then follow
    python main.py transcribe <input> <output> [lang]   # daemon-first, one-shot fallback
    python main.py models [base|small|medium|large-v3-turbo]  # fetch models from HF
    python main.py autostart on|off|status   # logon autostart (per-user, no admin)

Headless is the default: nothing on screen, no disturbance. `window` runs the
daemon in this console so you can see it working. No admin needed anywhere
(the daemon speaks localhost only).
"""
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER = HERE / "serve_transcribe.py"
CLIENT = HERE / "client.py"
FETCH = HERE / "scripts" / "fetch_models.py"
LOG = HERE / "daemon.log"
PORT_DEFAULT = 8764

from install_autostart import _venv_python as venv_python


def read_port() -> int:
    """Port from config.yaml, default 8764. Soft parser: control must work
    even when config is missing or mid-edit."""
    cfg = HERE / "config.yaml"
    if cfg.exists():
        for line in cfg.read_text(encoding="utf-8").splitlines():
            s = line.split("#")[0].strip()
            if s.startswith("port:"):
                try:
                    return int(s.split(":", 1)[1].strip())
                except ValueError:
                    pass
    return PORT_DEFAULT


def ask(req: dict, timeout: float = 5) -> dict:
    s = socket.create_connection(("127.0.0.1", read_port()), timeout=timeout)
    try:
        s.sendall((json.dumps(req) + "\n").encode())
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.split(b"\n")[0])
    finally:
        s.close()


def cmd_start() -> int:
    # detached: outlives this shell; log appends so history survives restarts
    logf = open(LOG, "ab")
    flags = 0
    if sys.platform == "win32":
        flags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    subprocess.Popen([str(venv_python()), str(SERVER)], stdout=logf,
                     stderr=subprocess.STDOUT, creationflags=flags, close_fds=True)
    # wait up to 120s: a blob-cache boot answers in ~2s, a first-ever boot
    # compiles the model (~1-2 min) and must not read as a false failure
    for _ in range(120):
        time.sleep(1)
        try:
            ask({"cmd": "ping"}, timeout=3)
            print(f"up: daemon serving on port {read_port()}")
            return 0
        except OSError:
            continue
    print("started, but not answering after 120s. A first boot compiles the model "
          "(~1-2 min, once). Watch it live:  python main.py log")
    return 1


def cmd_window() -> int:
    # window mode: same daemon, attached to this console, live log output
    print("udito daemon in this console (Ctrl+C stops it).")
    return subprocess.call([str(venv_python()), str(SERVER)])


def cmd_stop() -> int:
    try:
        ask({"cmd": "shutdown"})
    except OSError:
        print("down (nothing to stop)")
        return 0
    # wait until the port is truly free (SO_REUSEADDR keeps a dying socket briefly)
    for _ in range(20):
        time.sleep(0.5)
        try:
            ask({"cmd": "ping"}, timeout=2)
        except OSError:
            print("stopped (polite shutdown over the wire)")
            return 0
    print("shutdown sent but the port still answers; check 'python main.py log'")
    return 1


def cmd_restart() -> int:
    cmd_stop()
    return cmd_start()


def cmd_status() -> int:
    try:
        h = ask({"cmd": "health"})
    except OSError:
        print("down")
        return 3
    print(f"up: device={h.get('device')} model={h.get('model')} "
          f"uptime={h.get('uptime_s')}s concurrency={h.get('concurrency')} "
          f"emotion={h.get('emotion_model')} max_audio_s={h.get('max_audio_s')}")
    return 0


def cmd_log(n: int = 40) -> int:
    if not LOG.exists():
        print(f"no log yet at {LOG} (the daemon writes it on first start)")
        return 1
    lines = LOG.read_bytes().decode("utf-8", errors="replace").splitlines()
    for line in lines[-n:]:
        print(line)
    print("--- following (Ctrl+C to stop) ---")
    with open(LOG, "rb") as f:
        f.seek(0, 2)
        try:
            while True:
                line = f.readline()
                if line:
                    print(line.decode("utf-8", errors="replace").rstrip())
                else:
                    time.sleep(0.5)
        except KeyboardInterrupt:
            pass
    return 0


def main() -> None:
    argv = sys.argv[1:]
    cmd = argv[0].lower() if argv else "status"
    arg = argv[1] if len(argv) > 1 else None
    if cmd == "start":
        code = cmd_start()
    elif cmd == "window":
        code = cmd_window()
    elif cmd == "stop":
        code = cmd_stop()
    elif cmd == "restart":
        code = cmd_restart()
    elif cmd == "status":
        code = cmd_status()
    elif cmd == "log":
        code = cmd_log(int(arg) if arg and arg.isdigit() else 40)
    elif cmd == "transcribe":
        code = subprocess.call([str(venv_python()), str(CLIENT)] + argv[1:])
    elif cmd == "models":
        args = [str(venv_python()), str(FETCH)]
        if arg:
            args += ["--model", arg]
        code = subprocess.call(args)
    elif cmd == "autostart":
        import install_autostart as auto
        if arg == "on":
            auto.install()
            code = 0
        elif arg == "off":
            auto.remove()
            code = 0
        elif arg in ("status", None):
            auto._status()
            code = 0
        else:
            print("usage: python main.py autostart on|off|status")
            code = 2
    else:
        print(__doc__.strip())
        code = 2
    sys.exit(code)


if __name__ == "__main__":
    main()