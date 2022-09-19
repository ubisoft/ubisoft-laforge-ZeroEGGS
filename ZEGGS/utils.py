import numpy as np
from scipy import interpolate

from anim import bvh, quat


def change_bvh(filename, savename, order=None, fps=None, pace=1.0, center=False):
    anim_data = bvh.load(filename)
    output = anim_data.copy()

    if order is not None:
        output["order"] = order
        rotations = quat.unroll(quat.from_euler(np.radians(anim_data['rotations']), order=anim_data['order']))
        output["rotations"] = np.degrees(quat.to_euler(rotations, order=output["order"]))
    if pace is not None or fps is not None:
        if fps is None:
            fps = 1.0 / anim_data["frametime"]
        positions = anim_data['positions']
        rotations = quat.unroll(quat.from_euler(np.radians(anim_data['rotations']), order=anim_data['order']))
        nframes = positions.shape[0]
        nbones = positions.shape[1]
        original_times = np.linspace(0, nframes - 1, nframes)
        sample_times = np.linspace(
            0, nframes - 1, int(pace * (nframes * (fps * anim_data["frametime"]) - 1))
        )
        output["positions"] = interpolate.griddata(original_times, output["positions"].reshape([nframes, -1]),
                                                   sample_times, method='cubic').reshape([len(sample_times), nbones, 3])
        rotations = interpolate.griddata(original_times, rotations.reshape([nframes, -1]),
                                         sample_times, method='cubic').reshape([len(sample_times), nbones, 4])
        rotations = quat.normalize(rotations)
        output["rotations"] = np.degrees(quat.to_euler(rotations, order=output["order"]))
        output["frametime"] = 1.0 / fps

    if center:
        lrot = quat.from_euler(np.radians(output["rotations"]), output["order"])
        offset_pos = output["positions"][0:1, 0:1].copy() * np.array([1, 0, 1])
        offset_rot = lrot[0:1, 0:1].copy() * np.array([1, 0, 1, 0])

        root_pos = quat.mul_vec(quat.inv(offset_rot), output["positions"][:, 0:1] - offset_pos)
        output["positions"][:, 0:1] = quat.mul_vec(quat.inv(offset_rot),
                                                   output["positions"][:, 0:1] - offset_pos)
        output["rotations"][:, 0:1] = np.degrees(
            quat.to_euler(quat.mul(quat.inv(offset_rot), lrot[:, 0:1]), order=output["order"]))
    bvh.save(savename, output)


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
