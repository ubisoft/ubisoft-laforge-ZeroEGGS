"""Retarget BVH motion from the ZEGGS skeleton to the Geno character (GenoView, LaFAN1-resolved and friends).

Both skeletons have the same 75 joints and hierarchy; only bone lengths and rest orientations differ. Each joint's
Geno global rotation is its ZEGGS global rotation times a constant offset, and the root position is scaled per axis.
The offsets and scales are fitted on takes retargeted in Maya (https://github.com/orangeduck/zeroeggs-retarget).
"""
import argparse
from pathlib import Path

import numpy as np

from .anim import bvh, quat

PARAMS = Path(__file__).parent / "data" / "zeggs_to_geno.npz"


def _globals(anim):
    lrot = quat.unroll(quat.from_euler(np.radians(anim["rotations"]), anim["order"]))
    return quat.fk(lrot, anim["positions"], anim["parents"])


def _mean_quat(q):
    """Average of unit quaternions (rows), after putting them in one hemisphere."""
    q = q * np.sign(np.sum(q * q[:1], axis=-1, keepdims=True) + 1e-12)
    return quat.normalize(q.mean(0))


def fit(pairs, geno_skeleton):
    """Fit rotation offsets and root scaling from (zeggs_bvh, geno_bvh) path pairs of the same takes."""
    offsets, root_z, root_g = [], [], []
    for z_path, g_path in pairs:
        z, g = bvh.load(z_path), bvh.load(g_path)
        n = min(len(z["rotations"]), len(g["rotations"]))
        zr, zp = _globals(z)
        gr, gp = _globals(g)
        offsets.append(quat.mul(quat.inv(zr[:n]), gr[:n])[::10])
        root_z.append(zp[:n:10, 0])
        root_g.append(gp[:n:10, 0])
    offsets = np.concatenate(offsets)
    rotation_offset = np.stack([_mean_quat(offsets[:, j]) for j in range(offsets.shape[1])])
    root_z, root_g = np.concatenate(root_z), np.concatenate(root_g)
    # root: per-axis scale and shift, least squares
    scale, shift = np.zeros(3), np.zeros(3)
    for k in range(3):
        A = np.stack([root_z[:, k], np.ones(len(root_z))], 1)
        scale[k], shift[k] = np.linalg.lstsq(A, root_g[:, k], rcond=None)[0]
    skel = bvh.load(geno_skeleton)
    return dict(rotation_offset=rotation_offset, root_scale=scale, root_shift=shift,
                offsets=skel["offsets"], parents=skel["parents"], names=np.array(skel["names"]))


def retarget(anim, params=None):
    """ZEGGS-skeleton BVH dict to a Geno-skeleton BVH dict (same frames and frame time)."""
    p = params or dict(np.load(PARAMS))
    names = list(p["names"])
    assert list(anim["names"]) == names, "retarget expects the ZEGGS skeleton (same joint names and order as Geno)"
    grot, gpos = _globals(anim)
    geno_grot = quat.mul(grot, p["rotation_offset"])
    parents = p["parents"]
    lrot = geno_grot.copy()
    lrot[:, 1:] = quat.mul(quat.inv(geno_grot[:, parents[1:]]), geno_grot[:, 1:])
    lpos = np.repeat(p["offsets"][None].astype(np.float64), len(lrot), axis=0)
    lpos[:, 0] = gpos[:, 0] * p["root_scale"] + p["root_shift"]
    return dict(rotations=np.degrees(quat.to_euler(lrot, order="zyx")), positions=lpos, offsets=p["offsets"],
                parents=parents, names=names, order="zyx", frametime=anim["frametime"])


def joint_error(a, b):
    """Mean global joint-position error (BVH units) and mean global rotation error (degrees) between two clips."""
    ar, ap = _globals(a)
    br, bp = _globals(b)
    n = min(len(ap), len(bp))
    pos = np.linalg.norm(ap[:n] - bp[:n], axis=-1).mean()
    dot = np.abs(np.sum(ar[:n] * br[:n], axis=-1)).clip(0, 1)
    return float(pos), float(np.degrees(2 * np.arccos(dot)).mean())


def main():
    parser = argparse.ArgumentParser(description="Retarget ZEGGS-skeleton BVH files to the Geno character")
    parser.add_argument("input", nargs="+", help="ZEGGS-skeleton BVH file(s)")
    parser.add_argument("-o", "--output", help="Output file (one input) or folder (default: <input>_geno.bvh)")
    args = parser.parse_args()
    params = dict(np.load(PARAMS))
    for src in args.input:
        if args.output and len(args.input) == 1 and args.output.endswith(".bvh"):
            dst = Path(args.output)
        else:
            dst = (Path(args.output) if args.output else Path(src).parent) / (Path(src).stem + "_geno.bvh")
        dst.parent.mkdir(parents=True, exist_ok=True)
        bvh.save(dst, retarget(bvh.load(src), params))
        print(f"wrote {dst}")


if __name__ == "__main__":
    main()
