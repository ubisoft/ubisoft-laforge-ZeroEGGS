"""Foot-contact cleanup for BVH motion on the ZEGGS skeleton.

Planted feet are detected from ankle height and speed, pinned to one position and orientation per contact, and the
legs are re-solved with two-bone IK (hip and knee). If a pinned foot is out of reach, the hips are lowered instead of
over-stretching the leg. Units follow the BVH (cm for ZEGGS), with y up.
"""
import argparse

import numpy as np

from .anim import bvh, quat

LEGS = (("LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase"), ("RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"))


def _normalize(v, eps=1e-8):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + eps)


def _runs(mask):
    """[start, end) index pairs of the True runs in a 1D boolean array."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def detect_contacts(height, speed, height_thresh, speed_thresh, min_gap=5, min_contact=10):
    """Frames where a foot is planted: low and slow, with short gaps filled and short contacts dropped."""
    contact = (height < height_thresh) & (speed < speed_thresh)
    for s, e in _runs(~contact):
        if 0 < s and e < len(contact) and e - s < min_gap:
            contact[s:e] = True
    for s, e in _runs(contact):
        if e - s < min_contact:
            contact[s:e] = False
    return contact


def _split_runs(runs, pos, max_offset):
    """Split contacts whose raw foot wanders further than max_offset from the contact's mean position."""
    out = []
    for s, e in runs:
        if e - s >= 2 and np.linalg.norm(pos[s:e] - pos[s:e].mean(0), axis=-1).max() > max_offset:
            m = (s + e) // 2
            out += _split_runs([(s, m), (m, e)], pos, max_offset)
        else:
            out.append((s, e))
    return out


def _lock_targets(contact, pos, rot, blend, max_offset):
    """Per-frame pinned position, pinned rotation and blend weight (1 in contact, eased to 0 over `blend` frames)."""
    n = len(pos)
    weight = np.zeros(n)
    target_pos, target_rot = pos.copy(), rot.copy()
    best = np.full(n, np.inf)  # distance in frames to the nearest contact
    for s, e in _split_runs(_runs(contact), pos, max_offset):
        p = pos[s:e].mean(0)
        r = rot[s:e] * np.sign(np.sum(rot[s:e] * rot[s], axis=-1, keepdims=True))  # same hemisphere before averaging
        r = quat.normalize(r.mean(0))
        frames = np.arange(max(0, s - blend), min(n, e + blend))
        dist = np.maximum(np.maximum(s - frames, frames - (e - 1)), 0)
        closer = dist < best[frames]
        f, d = frames[closer], dist[closer]
        best[f] = d
        x = np.clip(1 - d / blend, 0, 1)
        weight[f] = x * x * (3 - 2 * x)  # smoothstep
        target_pos[f], target_rot[f] = p, r
    # Smooth the piecewise-constant targets so a split contact moves to its next pin over ~2*blend frames.
    target_rot *= np.sign(np.sum(target_rot * rot, axis=-1, keepdims=True) + 1e-12)
    target_pos, target_rot = _smooth(target_pos, blend), quat.normalize(_smooth(target_rot, blend))
    return target_pos, target_rot, weight


def _smooth(x, radius):
    """Hann-window moving average along time, with edge padding."""
    kernel = np.hanning(2 * radius + 3)[1:-1]
    kernel /= kernel.sum()
    padded = np.concatenate([x[:1].repeat(radius, 0), x, x[-1:].repeat(radius, 0)])
    return np.stack([np.convolve(padded[:, j], kernel, mode="valid") for j in range(x.shape[1])], axis=1)


def _two_bone_ik(a, b, c, t, a_gr, b_gr, a_lr, b_lr, eps=1e-5):
    """New local rotations of the hip (a) and knee (b) joints so the ankle (c) reaches t, keeping the knee's bend plane."""
    lab = np.linalg.norm(b - a, axis=-1)
    lcb = np.linalg.norm(b - c, axis=-1)
    lat = np.clip(np.linalg.norm(t - a, axis=-1), eps, lab + lcb - eps)

    def angle(u, v):
        return np.arccos(np.clip(np.sum(_normalize(u) * _normalize(v), axis=-1), -1, 1))

    ac_ab_0, ba_bc_0, ac_at_0 = angle(c - a, b - a), angle(a - b, c - b), angle(c - a, t - a)
    ac_ab_1 = np.arccos(np.clip((lcb * lcb - lab * lab - lat * lat) / (-2 * lab * lat), -1, 1))
    ba_bc_1 = np.arccos(np.clip((lat * lat - lab * lab - lcb * lcb) / (-2 * lab * lcb), -1, 1))

    axis0 = _normalize(np.cross(c - a, b - a))
    axis1 = _normalize(np.cross(c - a, t - a))
    r0 = quat.from_angle_axis(ac_ab_1 - ac_ab_0, quat.mul_vec(quat.inv(a_gr), axis0))
    r1 = quat.from_angle_axis(ba_bc_1 - ba_bc_0, quat.mul_vec(quat.inv(b_gr), axis0))
    r2 = quat.from_angle_axis(ac_at_0, quat.mul_vec(quat.inv(a_gr), axis1))
    return quat.mul(a_lr, quat.mul(r0, r2)), quat.mul(b_lr, r1)


def _contacts_and_velocity(gpos, names, fps, height_margin, speed_thresh):
    """Per ankle: contact mask and horizontal velocity (units/s, first frame 0)."""
    out = []
    for _, _, ankle, _ in LEGS:
        p = gpos[:, names.index(ankle)]
        speed = np.concatenate([[0.0], np.linalg.norm(np.diff(p, axis=0), axis=-1) * fps])
        contact = detect_contacts(p[:, 1], speed, np.percentile(p[:, 1], 5) + height_margin, speed_thresh)
        vel = np.concatenate([np.zeros((1, 2)), np.diff(p[:, [0, 2]], axis=0) * fps])
        out.append((contact, vel))
    return out


def fix_foot_sliding(anim, height_margin=3.0, speed_thresh=30.0, blend=15, max_offset=10.0, reach=0.995,
                     root_drift=True):
    """Return a copy of a loaded BVH dict with planted feet pinned and legs re-solved.

    Args:
        anim: dictionary from anim.bvh.load
        height_margin: an ankle within this distance above its resting height (5th percentile) can be in contact
        speed_thresh: an ankle slower than this (units per second) can be in contact
        blend: frames to ease the lock in and out
        max_offset: a contact is split where the raw foot wanders further than this from the pinned position
        reach: fraction of full leg length the IK may use before the hips are lowered
        root_drift: first remove whole-body skating, by subtracting the planted feet's horizontal drift from the root
    """
    names, parents, fps = anim["names"], anim["parents"], 1.0 / anim["frametime"]
    lrot = quat.unroll(quat.from_euler(np.radians(anim["rotations"]), anim["order"]))
    lpos = anim["positions"].copy()
    grot, gpos = quat.fk(lrot, lpos, parents)

    for _ in range(int(root_drift) * 2):  # second pass: contacts the raw drift hid become detectable
        # When the whole body skates, the planted feet move with the root. Moving the root back by the (smoothed)
        # average drift of the planted feet keeps the body where its feet are, so the foot lock below has little to do.
        drift, weight = np.zeros((len(lpos), 2)), np.zeros((len(lpos), 1))
        for contact, vel in _contacts_and_velocity(gpos, names, fps, height_margin, speed_thresh):
            drift += contact[:, None] * vel
            weight += contact[:, None]
        drift = _smooth(drift / np.maximum(weight, 1), blend)
        lpos[:, 0, [0, 2]] -= np.cumsum(drift, axis=0) / fps
        grot, gpos = quat.fk(lrot, lpos, parents)

    # Where each foot should be: raw pose blended towards its pinned contact pose.
    desired = []
    for hip, knee, ankle, _ in LEGS:
        i = names.index(ankle)
        p, r = gpos[:, i], grot[:, i]
        speed = np.concatenate([[0.0], np.linalg.norm(np.diff(p, axis=0), axis=-1) * fps])
        contact = detect_contacts(p[:, 1], speed, np.percentile(p[:, 1], 5) + height_margin, speed_thresh)
        tp, tr, w = _lock_targets(contact, p, r, blend, max_offset)
        tr = tr * np.sign(np.sum(tr * r, axis=-1, keepdims=True) + 1e-12)  # take the short way round
        # normalized lerp: quat.slerp divides by sin(angle) and returns zeros for identical inputs
        desired.append((p + w[:, None] * (tp - p), quat.normalize(r + w[:, None] * (tr - r))))

    # Lower the hips (smoothly) where a pinned foot is out of reach.
    drop = np.zeros(len(lpos))
    for (hip, knee, ankle, _), (dp, _) in zip(LEGS, desired):
        ia, ib, ic = (names.index(j) for j in (hip, knee, ankle))
        length = np.linalg.norm(gpos[:, ib] - gpos[:, ia], axis=-1) + np.linalg.norm(gpos[:, ic] - gpos[:, ib], axis=-1)
        drop = np.maximum(drop, np.linalg.norm(dp - gpos[:, ia], axis=-1) - reach * length)
    kernel = np.hanning(2 * blend + 1)
    drop = np.convolve(np.maximum(drop, 0), kernel / kernel.sum(), mode="same")
    lpos[:, 0, 1] -= drop
    grot, gpos = quat.fk(lrot, lpos, parents)

    # Re-solve each leg, then set the foot's local rotation so it keeps the desired global orientation.
    for (hip, knee, ankle, _), (dp, dr) in zip(LEGS, desired):
        ia, ib, ic = (names.index(j) for j in (hip, knee, ankle))
        a_lr, b_lr = _two_bone_ik(gpos[:, ia], gpos[:, ib], gpos[:, ic], dp, grot[:, ia], grot[:, ib],
                                  lrot[:, ia], lrot[:, ib])
        lrot[:, ia], lrot[:, ib] = a_lr, b_lr
        b_gr = quat.mul(quat.mul(grot[:, parents[ia]], a_lr), b_lr)
        lrot[:, ic] = quat.mul(quat.inv(b_gr), dr)

    out = dict(anim, positions=lpos, rotations=np.degrees(quat.to_euler(lrot, order=anim["order"])))
    return out


def foot_skating(anim, height_margin=3.0, moving=2.0):
    """Horizontal speed of feet that are near the floor, from foot height only (no contact labels).

    Returns mean speed (units/s) over low-foot frames and the fraction of those frames moving faster than `moving`.
    """
    names, fps = anim["names"], 1.0 / anim["frametime"]
    lrot = quat.from_euler(np.radians(anim["rotations"]), anim["order"])
    _, gpos = quat.fk(lrot, anim["positions"], anim["parents"])
    speeds = []
    for leg in LEGS:
        for joint in leg[2:]:
            p = gpos[:, names.index(joint)]
            low = (p[1:, 1] < np.percentile(p[:, 1], 5) + height_margin)
            v = np.linalg.norm(np.diff(p[:, [0, 2]], axis=0), axis=-1) * fps
            speeds.append(v[low])
    speeds = np.concatenate(speeds)
    return float(speeds.mean()), float((speeds > moving).mean())


def main():
    parser = argparse.ArgumentParser(description="Pin planted feet in a BVH (ZEGGS skeleton) and report foot skating")
    parser.add_argument("input", help="Input BVH")
    parser.add_argument("output", nargs="?", help="Output BVH (default: <input>_feet.bvh)")
    args = parser.parse_args()
    anim = bvh.load(args.input)
    fixed = fix_foot_sliding(anim)
    output = args.output or args.input.replace(".bvh", "_feet.bvh")
    bvh.save(output, fixed)
    for label, a in (("before", anim), ("after", fixed)):
        speed, frac = foot_skating(a)
        print(f"{label}: low-foot speed {speed:.2f} cm/s, moving {100 * frac:.1f}% of low-foot frames")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
