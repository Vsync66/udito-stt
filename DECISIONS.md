# Decision log

Design decisions worth keeping lineage for, in plain language. Code comments
state what the code does; this file records why it became that.

- 2026-10-07: Added RMS-based quiet-audio boost. Quiet takes made Whisper
  return empty text.
- 2026-10-08: Daemon-first architecture. Per-process pipeline compile took
  90+ seconds on NPU, so a resident daemon pays it once per boot.
- 2026-10-08: Boot-time config validation. A bad config must exit with the
  name of the file or key it rejected.
- 2026-10-08: Per-user autostart through the HKCU Run key. Task Scheduler
  logon tasks require admin rights; a shipped tool should not.
- 2026-10-09: Emotion classifier input normalized to zero mean and unit
  variance, matching the wav2vec2 training convention.
- 2026-10-09: Boot warmup. Lazy model loads settle after the socket is up.
- 2026-10-10: Punctuation reduced to one symbol. VAD-measured silences over
  a threshold become "...". Grammar restoration (neural period/capital
  insertion) was removed: a period claims a sentence boundary and can be
  wrong; "..." only claims the speaker paused.
- 2026-10-10: Noise reduction shipped as a three-way dial, default off.
  Live comparison on a clean recording: rnnoise left transcript and emotion
  unchanged; afftdn shifted a vowel reading and flipped the emotion label.
- 2026-10-10: Device auto-detection (NPU, then GPU, then CPU) so users
  without an NPU get a working daemon without editing config.
- 2026-10-10: Compile cache through the OpenVINO CACHE_DIR property,
  measured at 2.1s blob load versus 39-70s compile. Blobs are
  driver-specific, so the loader rebuilds the cache when a blob fails to
  load (one cold boot after each driver update).
- 2026-10-10: Config parser strips trailing comments from quoted values.
  Found by the boot gate: `key: "" # comment` parsed as a garbage path.
- 2026-10-10: One control entry point. `main.py start/window/stop/restart/
  status/log` runs everything; headless is the default and `window` keeps
  the live log for people who want to see the daemon work.
- 2026-10-10: Silence returns without emotion. A zero-energy take skips the
  emotion head too: the tag would be a model artifact, not a reading.