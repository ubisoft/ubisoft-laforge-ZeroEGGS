"""Generation must match the original ZEGGS code (outputs recorded before the refactor)."""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from zeggs.anim import bvh
from zeggs.generate import generate_gesture

DATA = Path(__file__).resolve().parents[1] / "data"
EXAMPLE = DATA / "clean" / "067_Speech_2_x_1_0.bvh"
pytestmark = pytest.mark.skipif(not EXAMPLE.exists(), reason="ZEGGS dataset not extracted into data/")


@pytest.fixture(scope="module")
def reference():
    return np.load(Path(__file__).parent / "reference" / "original_code_outputs.npz")


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    # first 20 s of the take, as in the recorded reference
    fs, x = wavfile.read(EXAMPLE.with_suffix(".wav"))
    path = tmp_path_factory.mktemp("audio") / "clip.wav"
    wavfile.write(path, fs, x[: fs * 20])
    return path


def check(out_bvh, reference, version):
    out = bvh.load(out_bvh)
    # Differences come only from float rounding (positional table recomputed on this machine).
    np.testing.assert_allclose(out["positions"][::10], reference[f"{version}_positions"], atol=1e-3)
    np.testing.assert_allclose(out["rotations"][::10], reference[f"{version}_rotations"], atol=1e-2)


def test_example_based_v1(tmp_path, clip, reference):
    generate_gesture(clip, [(EXAMPLE, (0, 600))], DATA / "outputs/v1/saved_models", DATA / "processed_v1",
                     tmp_path, file_name="v1", use_gpu=False)
    check(tmp_path / "v1.bvh", reference, "v1")


def test_label_based_v2(tmp_path, clip, reference):
    generate_gesture(clip, ["Happy"], DATA / "outputs/v2/saved_models", DATA / "processed_v2", tmp_path,
                     style_encoding_type="label", first_pose=str(EXAMPLE), use_gpu=False)
    check(tmp_path / "audio_clip_label_Happy.bvh", reference, "v2")


def test_style_embedding_only(tmp_path):
    embedding = generate_gesture(None, [(EXAMPLE, (0, 600))], DATA / "outputs/v1/saved_models",
                                 DATA / "processed_v1", None, use_gpu=False)
    assert embedding.shape == (1, 64)
