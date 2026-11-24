from pathlib import Path

import os

import numpy as np
import pytest

from zeggs.anim import bvh
from zeggs.retarget import PARAMS, joint_error, retarget

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "data" / "clean" / "067_Speech_2_x_1_0.bvh"


@pytest.mark.skipif(not EXAMPLE.exists(), reason="ZEGGS dataset not extracted into data/")
def test_retarget_gives_geno_skeleton_and_keeps_timing():
    z = bvh.load(EXAMPLE, start=0, end=120)
    g = retarget(z)
    geno = np.load(PARAMS)
    assert g["names"] == list(geno["names"]) and len(g["rotations"]) == len(z["rotations"])
    np.testing.assert_allclose(g["offsets"][1:], geno["offsets"][1:])  # bone lengths of the Geno mesh
    assert np.isfinite(g["rotations"]).all() and g["frametime"] == z["frametime"]


def test_retarget_matches_maya_reference_when_available():
    """Against takes retargeted in Maya (zeroeggs-retarget), with that file's bone lengths swapped in."""
    # set ZEGGS_GENO_REF to the extracted `bvh` folder of https://github.com/orangeduck/zeroeggs-retarget
    ref = Path(os.environ.get("ZEGGS_GENO_REF", "missing")) / "065_Speech_0_x_1_0.bvh"
    src = ROOT / "data" / "clean" / ref.name
    if not (ref.exists() and src.exists()):
        pytest.skip("set ZEGGS_GENO_REF to the zeroeggs-retarget bvh folder")
    g_ref = bvh.load(ref, start=0, end=600)
    out = retarget(bvh.load(src, start=0, end=600))
    out["positions"][:, 1:] = g_ref["offsets"][1:]
    pos, rot = joint_error(out, g_ref)
    assert pos < 0.05 and rot < 0.1
