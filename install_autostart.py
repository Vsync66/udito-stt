"""
install_autostart.py, set up the udito daemon autostart.

For shipped users: download udito, configure config.yaml, run this once.
The daemon then starts at every logon (hidden, silent), no more 90-225s compile
paid per session; the compile is paid once per boot and amortized all day.

Usage:
  python install_autostart.py            # install (or update) the logon hook
  python install_autostart.py --remove   # uninstall it
  python install_autostart.py --status   # show whether it's installed

How it works (Windows): registers a per-user HKCU Run entry that runs
run_daemon_silent.vbs, which launches the venv python on serve_transcribe.py
with zero visible windows. Logs land in daemon.log.
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK_NAME = "udito_daemon"

VBS_NAME = "run_daemon_silent.vbs"
LAUNCHER_BAT = "start_daemon.bat"


def _venv_python() -> Path:
    """The daemon python: npu_lab venv on this layout, sibling .venv for shipped layout,
    bare python as last resort (fail-visible note printed)."""
    env_override = os.environ.get("UDITO_PYTHON")
    candidates = ([Path(env_override)] if env_override else []) + [
        HERE.parent / ".venv" / "Scripts" / "python.exe",   # shipped: udito under a venv-parent
        HERE / ".venv" / "Scripts" / "python.exe"]          # self-contained venv inside the folder
    for c in candidates:
        if c.exists():
            return c
    print("[autostart] WARNING: no venv python found next to udito; using bare 'python'.\n"
          "           If the daemon fails to import openvino, create a venv with openvino installed.", file=sys.stderr)
    return Path(sys.executable)


def _write_launchers(py: Path) -> None:
    bat = HERE / LAUNCHER_BAT
    bat.write_text(
        "@echo off\r\n"
        f'"{py}" "{HERE / "serve_transcribe.py"}" >> "{HERE / "daemon.log"}" 2>&1\r\n',
        encoding="utf-8")
    vbs = HERE / VBS_NAME
    vbs.write_text(
        'Set sh = CreateObject("WScript.Shell")\n'
        f'sh.Run """{bat}""", 0, False\n',
        encoding="utf-8")
    print(f"[autostart] launchers written: {LAUNCHER_BAT}, {VBS_NAME}")


def _schtasks(args: list) -> subprocess.CompletedProcess:
    return subprocess.run(["schtasks"] + args, capture_output=True, text=True)


def install() -> None:
    py = _venv_python()
    _write_launchers(py)
    # Per-user logon hook: HKCU Run key. No admin required (schtasks ONLOGON needs
    # elevation, and a shipped tool shouldn't demand admin).
    import winreg
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    value = f'wscript.exe "{HERE / VBS_NAME}"'
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE) as k:
            winreg.SetValueEx(k, TASK_NAME, 0, winreg.REG_SZ, value)
    except OSError as e:
        print(f"[autostart] INSTALL FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"[autostart] installed: '{TASK_NAME}' in HKCU Run, daemon starts at every logon (this user).")
    print("[autostart] first boot after install pays the compile once (~70-225s); "
          "after that, no session ever pays it again.")
    _status()


def remove() -> None:
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Run",
                            0, winreg.KEY_SET_VALUE) as k:
            winreg.DeleteValue(k, TASK_NAME)
        print(f"[autostart] '{TASK_NAME}' removed from HKCU Run.")
    except FileNotFoundError:
        print(f"[autostart] '{TASK_NAME}' not installed (nothing to remove).")


def _status() -> None:
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run") as k:
            winreg.QueryValueEx(k, TASK_NAME)
        print("[autostart] status: INSTALLED (HKCU Run)")
    except FileNotFoundError:
        print("[autostart] status: not installed")


if __name__ == "__main__":
    if "--remove" in sys.argv:
        remove()
    elif "--status" in sys.argv:
        _status()
    else:
        install()