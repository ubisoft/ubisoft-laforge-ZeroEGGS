import numpy as np

from .anim import bvh, quat


def write_bvh(
        filename,
        V_root_pos,
        V_root_rot,
        V_lpos,
        V_lrot,
        parents,
        names,
        order,
        dt,
        start_position=None,
        start_rotation=None,
):
    if start_position is not None and start_rotation is not None:
        offset_pos = V_root_pos[0:1].copy()
        offset_rot = V_root_rot[0:1].copy()

        V_root_pos = quat.mul_vec(quat.inv(offset_rot), V_root_pos - offset_pos)
        V_root_rot = quat.mul(quat.inv(offset_rot), V_root_rot)
        V_root_pos = (
                quat.mul_vec(start_rotation[np.newaxis], V_root_pos) + start_position[np.newaxis]
        )
        V_root_rot = quat.mul(start_rotation[np.newaxis], V_root_rot)

    V_lpos = V_lpos.copy()
    V_lrot = V_lrot.copy()
    V_lpos[:, 0] = quat.mul_vec(V_root_rot, V_lpos[:, 0]) + V_root_pos
    V_lrot[:, 0] = quat.mul(V_root_rot, V_lrot[:, 0])

    bvh.save(
        filename,
        dict(
            order=order,
            offsets=V_lpos[0],
            names=names,
            frametime=dt,
            parents=parents,
            positions=V_lpos,
            rotations=np.degrees(quat.to_euler(V_lrot, order=order)),
        ),
    )
