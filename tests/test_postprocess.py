from pathlib import Path

import numpy as np
import pytest

from zeggs.anim import bvh, quat
from zeggs.postprocess import LEGS, _two_bone_ik, detect_contacts, fix_foot_sliding, foot_skating

EXAMPLE = Path(__file__).resolve().parents[1] / "data" / "clean" / "067_Speech_2_x_1_0.bvh"
needs_data = pytest.mark.skipif(not EXAMPLE.exists(), reason="ZEGGS dataset not extracted into data/")


def load_clip(frames=900):
    a = bvh.load(EXAMPLE)
    a["rotations"], a["positions"] = a["rotations"][:frames], a["positions"][:frames]
    return a


def globals_of(a):
    return quat.fk(quat.from_euler(np.radians(a["rotations"]), a["order"]), a["positions"], a["parents"])


def test_detect_contacts_fills_gaps_and_drops_blips():
    height = np.zeros(40)
    speed = np.zeros(40)
    speed[10:12] = 99  # short gap inside a contact: filled
    speed[20:38] = 99  # long gap: kept
    speed[30] = 0  # one-frame "contact" inside the gap: dropped
    contact = detect_contacts(height, speed, 1.0, 1.0)
    assert contact[:20].all() and not contact[20:38].any()


@needs_data
def test_two_bone_ik_reaches_reachable_targets():
    a = load_clip(300)
    lrot = quat.from_euler(np.radians(a["rotations"]), a["order"])
    grot, gpos = quat.fk(lrot, a["positions"], a["parents"])
    ia, ib, ic = (a["names"].index(j) for j in LEGS[0][:3])
    # targets well inside reach: pull the ankle 5 cm towards the hip
    t = gpos[:, ic] + 5 * (gpos[:, ia] - gpos[:, ic]) / np.linalg.norm(gpos[:, ia] - gpos[:, ic], axis=-1, keepdims=True)
    lrot[:, ia], lrot[:, ib] = _two_bone_ik(gpos[:, ia], gpos[:, ib], gpos[:, ic], t, grot[:, ia], grot[:, ib],
                                            lrot[:, ia], lrot[:, ib])
    _, new = quat.fk(lrot, a["positions"], a["parents"])
    assert np.linalg.norm(new[:, ic] - t, axis=-1).max() < 0.5


@needs_data
def test_fix_reduces_skating_and_keeps_upper_body():
    a = load_clip()
    # add a 3 cm side-to-side wobble of the whole body, which makes the planted feet slide
    a["positions"] = a["positions"].copy()
    tt = np.arange(len(a["positions"])) / 60
    a["positions"][:, 0, 0] += 3 * np.sin(2 * np.pi * 0.3 * tt)
    fixed = fix_foot_sliding(a)
    before, after = foot_skating(a)[0], foot_skating(fixed)[0]
    assert after < 0.5 * before
    g0, g1 = globals_of(a)[1], globals_of(fixed)[1]
    head = a["names"].index("Head")
    # only the hips may move (vertically, when a foot is out of reach), so the head shifts by at most that drop
    drop = np.abs(fixed["positions"][:, 0, 1] - a["positions"][:, 0, 1]).max()
    assert np.abs(g1[:, head] - g0[:, head]).max() <= drop + 1e-3


@needs_data
def test_still_pose_is_unchanged():
    a = load_clip(200)
    a["rotations"] = a["rotations"][:1].repeat(200, 0)
    a["positions"] = a["positions"][:1].repeat(200, 0)
    fixed = fix_foot_sliding(a)
    np.testing.assert_allclose(globals_of(fixed)[1], globals_of(a)[1], atol=1e-3)
