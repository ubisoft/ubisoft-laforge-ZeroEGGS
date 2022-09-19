import numpy as np

from . import mat


def to_translation(x):
    return x[..., :3, 3] / x[..., 3, 3][..., np.newaxis]


def to_rotation(x):
    return x[..., :3, :3]


def to_rotation_translation(x):
    return to_rotation(x), to_translation(x)


def log(x, eps=1e-10):
    angle, axis = to_angle_axis(x, eps=eps)
    return (angle / 2.0)[..., np.newaxis] * axis


def exp(x, eps=1e-10):
    halfangle = np.sqrt(np.sum(x ** 2.0, axis=-1))
    axis = x[..., :3] / (halfangle[..., np.newaxis] + eps)
    return from_angle_axis(2.0 * halfangle, axis)


def to_angle_axis(x, eps=1e-10):
    angle = np.arccos(np.clip((x[..., 0, 0] + x[..., 1, 1] + x[..., 2, 2] - 1.0) / 2.0, 0.0, 1.0))
    axis = np.concatenate([
        (x[..., 2, 1] - x[..., 1, 2])[..., np.newaxis],
        (x[..., 0, 2] - x[..., 2, 0])[..., np.newaxis],
        (x[..., 1, 0] - x[..., 0, 1])[..., np.newaxis]
    ], axis=-1) / ((2.0 * np.sin(angle))[..., np.newaxis] + eps)

    return angle, axis


def from_rotation_translation(rot, pos):
    x = np.concatenate([rot, pos[..., np.newaxis]], axis=-1)
    x = np.concatenate([x, np.ones_like(x)[..., :1, :] * np.array([0, 0, 0, 1], dtype=np.float32)], axis=-2)
    return x


def from_angle_axis(angle, axis):
    angle = angle[..., np.newaxis]
    a0, a1, a2 = axis[..., 0:1], axis[..., 1:2], axis[..., 2:3]
    c, s, t = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)

    r0 = np.concatenate([c + a0 * a0 * t, a0 * a1 * t - a2 * s, a0 * a2 * t + a1 * s], axis=-1)
    r1 = np.concatenate([a0 * a1 * t + a2 * s, c + a1 * a1 * t, a1 * a2 * t - a0 * s], axis=-1)
    r2 = np.concatenate([a0 * a2 * t - a1 * s, a1 * a2 * t + a0 * s, c + a2 * a2 * t], axis=-1)

    return np.concatenate([r0[..., np.newaxis, :], r1[..., np.newaxis, :], r2[..., np.newaxis, :]], axis=-2)


def from_euler(e, order='zyx'):
    c, s = np.cos(e), np.sin(e)
    c0, c1, c2 = c[..., 0:1], c[..., 1:2], c[..., 2:3]
    s0, s1, s2 = s[..., 0:1], s[..., 1:2], s[..., 2:3]

    if order == 'xzy':
        r0 = np.concatenate([c1 * c2, -s1, c1 * s2], axis=-1)
        r1 = np.concatenate([s0 * s2 + c0 * c2 * s1, c0 * c1, c0 * s1 * s2 - c2 * s0], axis=-1)
        r2 = np.concatenate([c2 * s0 * s1 - c0 * s2, c1 * s0, c0 * c2 + s0 * s1 * s2], axis=-1)
    elif order == 'xyz':
        r0 = np.concatenate([c1 * c2, -c1 * s2, s1], axis=-1)
        r1 = np.concatenate([c0 * s2 + c2 * s0 * s1, c0 * c2 - s0 * s1 * s2, -c1 * s0], axis=-1)
        r2 = np.concatenate([s0 * s2 - c0 * c2 * s1, c2 * s0 + c0 * s1 * s2, c0 * c1], axis=-1)
    elif order == 'yxz':
        r0 = np.concatenate([c0 * c2 + s0 * s1 * s2, c2 * s0 * s1 - c0 * s2, c1 * s0], axis=-1)
        r1 = np.concatenate([c1 * s2, c1 * c2, -s1], axis=-1)
        r2 = np.concatenate([c0 * s1 * s2 - c2 * s0, c0 * c2 * s1 + s0 * s2, c0 * c1], axis=-1)
    elif order == 'yzx':
        r0 = np.concatenate([c0 * c1, s0 * s2 - c0 * c2 * s1, c2 * s0 + c0 * s1 * s2], axis=-1)
        r1 = np.concatenate([s1, c1 * c2, -c1 * s2], axis=-1)
        r2 = np.concatenate([-c1 * s0, c0 * s2 + c2 * s0 * s1, c0 * c2 - s0 * s1 * s2], axis=-1)
    elif order == 'zyx':
        r0 = np.concatenate([c0 * c1, c0 * s1 * s2 - c2 * s0, s0 * s2 + c0 * c2 * s1], axis=-1)
        r1 = np.concatenate([c1 * s0, c0 * c2 + s0 * s1 * s2, c2 * s0 * s1 - c0 * s2], axis=-1)
        r2 = np.concatenate([-s1, c1 * s2, c1 * c2], axis=-1)
    elif order == 'zxy':
        r0 = np.concatenate([c0 * c2 - s0 * s1 * s2, -c1 * s0, c0 * s2 + c2 * s0 * s1], axis=-1)
        r1 = np.concatenate([c2 * s0 + c0 * s1 * s2, c0 * c1, s0 * s2 - c0 * c2 * s1], axis=-1)
        r2 = np.concatenate([-c1 * s2, s1, c1 * c2], axis=-1)
    else:
        raise Exception('Unknown ordering: %s' % order)

    return np.concatenate([r0[..., np.newaxis, :], r1[..., np.newaxis, :], r2[..., np.newaxis, :]], axis=-2)


def from_basis(x, y, z):
    return np.concatenate([x[..., np.newaxis], y[..., np.newaxis], z[..., np.newaxis]], axis=-1)


def orthogonalize(x, method='svd', eps=0.0):
    def cross(a, b):
        return np.concatenate([
            a[..., 1:2] * b[..., 2:3] - a[..., 2:3] * b[..., 1:2],
            a[..., 2:3] * b[..., 0:1] - a[..., 0:1] * b[..., 2:3],
            a[..., 0:1] * b[..., 1:2] - a[..., 1:2] * b[..., 0:1],
        ], axis=-1)

    if method == 'cross':
        r0, r1 = x[..., 0], x[..., 1]
        r2 = cross(r0, r1)
        r0 = r0 / (np.sqrt(np.sum(r0 * r0, axis=-1))[..., np.newaxis] + eps)
        r2 = r2 / (np.sqrt(np.sum(r2 * r2, axis=-1))[..., np.newaxis] + eps)
        r1 = cross(r2, r0)
        return from_basis(r0, r1, r2)
    elif method == 'svd':
        U, _, V = mat.svd(x)
        return mat.mul(U, V)
    else:
        raise ValueError('Unknown method \'%s\'' % method)


def orthogonalize_from_xy(xy):
    xaxis = xy[..., 0:1, :]
    zaxis = np.cross(xaxis, xy[..., 1:2, :])
    yaxis = np.cross(zaxis, xaxis)

    output = np.concatenate([
        xaxis / np.sqrt(np.sum(xaxis * xaxis, axis=-1))[..., np.newaxis],
        yaxis / np.sqrt(np.sum(yaxis * yaxis, axis=-1))[..., np.newaxis],
        zaxis / np.sqrt(np.sum(zaxis * zaxis, axis=-1))[..., np.newaxis]
    ], axis=-2)

    return mat.transpose(output)


def orthogonalize_iterative(ts, R=None, iterations=4, eps=1e-5):
    if R is None:
        R = np.zeros_like(ts)
        R[..., :, :] = np.eye(3)

    for _ in range(iterations):
        omega = ((
                         np.cross(R[..., :, 0], ts[..., :, 0]) +
                         np.cross(R[..., :, 1], ts[..., :, 1]) +
                         np.cross(R[..., :, 2], ts[..., :, 2])
                 ) / (abs(
            np.sum(R[..., :, 0] * ts[..., :, 0], axis=-1)[..., np.newaxis] +
            np.sum(R[..., :, 1] * ts[..., :, 1], axis=-1)[..., np.newaxis] +
            np.sum(R[..., :, 2] * ts[..., :, 2], axis=-1)[..., np.newaxis]
        ) + eps))

        w = np.sqrt(np.sum(np.square(omega), axis=-1))

        R = mat.mul(from_angle_axis(w, omega / (w[..., np.newaxis] + eps)), R)

    return R
