# Connecting Udito to agents

Udito is a localhost TCP daemon speaking line-delimited JSON. Anything that can
open a socket can use it. This file shows the protocol, then the exact wiring
for Hermes (and anything shaped like it).

## The protocol

One request per line, one response per line, port 8764 (config):

```json
{"cmd": "ping"}
{"cmd": "health"}
{"cmd": "transcribe", "path": "C:/audio/note.webm", "lang": "en"}
{"cmd": "unload"}
{"cmd": "shutdown"}
```

A transcribe response:

```json
{
  "ok": true,
  "text": "Alright, Nico. This file is gonna serve ... as the testing file ... [calm 0.84]",
  "elapsed": 1.35,
  "audio_dur": 70.81,
  "punct": "pause",
  "hints": {"gaps": 19, "mode": "ellipsis-vad"},
  "emotion": {"ok": true, "emotion": "calm", "confidence": 0.84},
  "pipe": "NPU"
}
```

Request fields on transcribe:

| field | meaning | default |
|-------|---------|---------|
| `path` | audio file path, any format ffmpeg reads (webm, mp3, m4a, wav) | required |
| `lang` | ISO code like `en`, empty = auto-detect | config |
| `punct` | `off` (raw) or `pause` (`...` from real silence) | `pause` |
| `emotion` | true = append `[emotion score]` to the text + full breakdown | false |
| `noise_reduction` | `off` / `afftdn` / `rnnoise` for this request only | config |

Errors come back as `{"ok": false, "error": "<the problem and the fix>"}`.
The daemon never crashes on a bad request; it answers with the reason.
`unload` drops whisper + emotion from RAM without killing the daemon; the
next transcribe rebuilds from the blob cache in about 2 seconds.

## Wiring into Hermes

Hermes calls an STT provider with `{input_path} {output_path} {language}`.
Udito ships `client.py` for exactly this shape: try the daemon, fall back to
one-shot if it's down, write the transcript to the output path.

In the Hermes profile's `config.yaml`:

```yaml
stt:
  provider: udito
  normalize: true
  providers:
    udito:
      type: command
      command: "C:/path/to/udito/udito.bat {input_path} {output_path} {language}"
```

Notes earned the hard way:

- The block goes under `stt.providers.<name>`, not top-level `providers.*`.
  Hermes silently ignores an unknown top-level namespace.
- `normalize: true` makes Hermes pre-convert webm to 16kHz wav. Udito decodes
  again on its own side, so this is belt and suspenders; Hermes's own ffmpeg
  may not exist on PATH. Both sides being able to decode is the safe default.
- Voice-recording shape (webm opus) arrives at the daemon and leaves as text
  in under 2 seconds warm.

## Wiring into anything else

The contract is three lines of Python:

```python
import socket, json
def udito(path, lang="en"):
    s = socket.create_connection(("127.0.0.1", 8764), timeout=300)
    s.sendall((json.dumps({"cmd": "transcribe", "path": path, "lang": lang}) + "\n").encode())
    buf = b""
    while b"\n" not in buf:
        buf += s.recv(65536)
    return json.loads(buf.split(b"\n")[0])
```

Long-running callers: retry once on connection-refused (the daemon may be
mid-boot), and treat `ok: false` as a final semantic answer rather than a
transport failure. The one-shot fallback (`run_transcribe.py`) exists for when
the daemon is genuinely down, and costs a full model compile, so prefer fixing
the daemon.

## Lifecycle

Everything runs through `main.py`:

```powershell
python main.py start        # boot headless: no window, log -> daemon.log
python main.py window       # boot in this console: live log, Ctrl+C stops
python main.py stop         # polite shutdown over the wire
python main.py restart
python main.py status       # device, model, uptime, emotion residency
python main.py log          # tail daemon.log
python main.py autostart on # start at logon, no admin
```

The `health` command reports device, model, uptime, concurrency, and whether
the emotion head is resident. It is the monitoring hook for supervisors.