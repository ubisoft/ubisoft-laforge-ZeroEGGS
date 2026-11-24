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
    # the upper body keeps its pose: only leg joints and the root may change
    legs = {a["names"].index(j) for leg in LEGS for j in leg[:3]} | {0}
    others = [i for i in range(len(a["names"])) if i not in legs]
    r0 = quat.from_euler(np.radians(a["rotations"]), a["order"])[:, others]
    r1 = quat.from_euler(np.radians(fixed["rotations"]), fixed["order"])[:, others]
    np.testing.assert_allclose(np.abs(np.sum(r0 * r1, axis=-1)), 1.0, atol=1e-5)


def test_root_drift_removes_whole_body_skating():
    """A body gliding sideways at 10 cm/s with still feet: the root correction should remove the glide."""
    if not EXAMPLE.exists():
        pytest.skip("ZEGGS dataset not extracted into data/")
    a = load_clip(600)
    a["rotations"] = a["rotations"][:1].repeat(600, 0)
    a["positions"] = a["positions"][:1].repeat(600, 0)
    a["positions"][:, 0, 0] += 10 * np.arange(600) / 60
    assert foot_skating(a)[0] > 5
    assert foot_skating(fix_foot_sliding(a))[0] < 0.5
    assert foot_skating(fix_foot_sliding(a, root_drift=False))[0] > foot_skating(fix_foot_sliding(a))[0]


@needs_data
def test_still_pose_is_unchanged():
    a = load_clip(200)
    a["rotations"] = a["rotations"][:1].repeat(200, 0)
    a["positions"] = a["positions"][:1].repeat(200, 0)
    fixed = fix_foot_sliding(a)
    np.testing.assert_allclose(globals_of(fixed)[1], globals_of(a)[1], atol=1e-3)


@needs_data
def test_heading_drift_is_limited_and_small_turns_kept():
    from zeggs.postprocess import fix_heading_drift, heading
    a = load_clip(3600)
    # add a steady 2 deg/s turn of the whole body (120 deg over the minute), about the root
    turn = quat.from_angle_axis(np.radians(2.0 * np.arange(3600) / 60), np.array([0.0, 1.0, 0.0]))
    lrot = quat.from_euler(np.radians(a["rotations"]), a["order"])
    lrot[:, 0] = quat.mul(turn, lrot[:, 0])
    drifting = dict(a, rotations=np.degrees(quat.to_euler(lrot, order=a["order"])))
    yaw = heading(fix_heading_drift(drifting))
    assert np.abs(heading(drifting) - heading(drifting)[0]).max() > 100
    assert np.abs(yaw - yaw[0]).max() < 50  # 20 deg trend limit plus the clip's own faster turns
    # motion that stays within the limit is left alone
    np.testing.assert_allclose(globals_of(fix_heading_drift(a))[1], globals_of(a)[1], atol=1e-3)
