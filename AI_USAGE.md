# AI Usage

Udito was designed by Vihaan Kodiganti. Every product decision is his: the
overall architecture (an NPU-resident transcription daemon with a lazy-loaded
emotion pipe), the one-symbol speech pause as the project's punctuation law,
emotion on OpenVINO as a design goal, and the choice to lazy-load heavy
components. The decision log in DECISIONS.md records these choices.

The implementation was built and debugged in pair sessions between Vihaan and
Nico di Angelo, an AI agent. Quantization decisions came out of that pair
work, and some were Vihaan's own findings, including the insight that INT8
made long recordings feasible on the NPU.

Nova Washington, a second AI agent acting as peer reviewer, did the ship-gate review and
a later audit pass. That review found real defects: dead code, a
boot-validation bypass, and a word-duplication bug in the sliding-window
stitcher. All were fixed before release.

The benchmarks in this repository were executed and verified against the
running system, not estimated by any AI. Code comments describe what the code
does. No AI generated benchmark numbers, design decisions, or claims in this
repo.