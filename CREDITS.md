# Credits

Udito stands on other people's work. All of it attributed here.

- **Whisper** by OpenAI (MIT license). The transcription model.
- **OpenVINO** and **OpenVINO GenAI** by Intel (Apache-2.0). Inference
  runtime and the Whisper pipeline implementation.
- **RNNoise reference model** (`models/rnnoise/std.rnnn`, ~300KB) from the
  rnnoise-models collection by Gregor Richards, based on Xiph.Org's RNNoise
  (public domain, per the collection's statement).
- **wav2vec2** by META research (MIT). The emotion classifier's architecture,
  used here through a community ONNX export.
- **ffmpeg** (LGPL/GPL builds). Audio decoding and the afftdn/arnndn filters.

Design and implementation: Vihaan Kodiganti (design authority) with Nico
di Angelo. The decision log in DECISIONS.md records why each design choice
became what it is.