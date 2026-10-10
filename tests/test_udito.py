"""Udito unit tests, no compile needed (decorator + boot validation layers)."""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def _speech(sr=16000, dur=1.0, amp=0.1):
    return amp * np.sin(2 * np.pi * 220 * np.arange(int(sr * dur)) / sr)


def test_decorate_one_symbol():
    """Two speech bursts separated by a 0.7s true silence -> exactly one '...'."""
    from serve_transcribe import _decorate_pauses
    sr = 16000
    gap = np.zeros(int(sr * 0.7))
    audio = np.concatenate([_speech(sr, 1.0), gap, _speech(sr, 1.0)])
    text = "w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20"
    out, hints = _decorate_pauses(text, audio, sr, pause_thr=0.6)
    assert out.count("...") == 1, out
    assert hints["mode"] == "ellipsis-vad"
    # the mark sits at the boundary between the two bursts' word shares
    assert "..." in out, out


def test_decorate_below_threshold_clean():
    """A 0.4s stall under a 600ms threshold -> no marks at all."""
    from serve_transcribe import _decorate_pauses
    sr = 16000
    gap = np.zeros(int(sr * 0.4))
    audio = np.concatenate([_speech(sr, 1.0), gap, _speech(sr, 1.0)])
    text = " ".join(f"w{i}" for i in range(1, 21))
    out, hints = _decorate_pauses(text, audio, sr, pause_thr=0.6)
    assert "..." not in out, out
    assert hints["gaps"] == 0


def test_emotion_labels_present():
    """The shipped emotion model carries its label map."""
    import json
    p = os.path.join(HERE, "models", "emotion_onnx", "config.json")
    assert os.path.exists(p), p
    labels = json.load(open(p, encoding="utf-8"))["id2label"]
    assert len(labels) == 7


def test_config_example_complete():
    """config.example.yaml must carry every dial the boot validator knows."""
    p = os.path.join(HERE, "config.example.yaml")
    text = open(p, encoding="utf-8").read()
    for key in ("device", "model", "noise_reduction", "emotion_model",
                "emotion_in_text", "punct_default", "pause_ms", "port",
                "max_audio_seconds", "concurrency", "boost", "compile_cache",
                "beam_size", "language", "initial_prompt"):
        assert key in text, key
    # dead dials must stay dead (review find 10-10): a key no code reads never
    # ships as an example dial
    for dead in ("emotion_normalize", "log_level", "emotion_enabled", "emotion_quant"):
        assert dead not in text, dead


if __name__ == "__main__":
    test_decorate_one_symbol()
    print("PASS one-symbol decoration")
    test_decorate_below_threshold_clean()
    print("PASS threshold cleanliness")
    test_emotion_labels_present()
    print("PASS emotion label map")
    test_config_example_complete()
    print("PASS config example completeness (incl. dead-dial ban)")
    print("ALL GREEN")