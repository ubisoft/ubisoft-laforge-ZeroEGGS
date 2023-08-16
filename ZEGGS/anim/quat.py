import numpy as np


def eye(shape=[]):
    if shape == []:
        return np.array([1, 0, 0, 0], dtype=np.float32)
    else:
        return np.array([1, 0, 0, 0], dtype=np.float32) * np.ones(
            np.concatenate([shape, [4]], axis=0), dtype=np.float32)


def eye_like(x):
    return np.array([1, 0, 0, 0], dtype=np.float32) * np.ones_like(
        x[..., np.newaxis].repeat(4, axis=-1))


def mul(x, y):
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    return np.concatenate([
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0], axis=-1)


def _fast_cross(a, b):
    o = np.empty(np.broadcast(a, b).shape)
    o[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    o[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    o[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return o


def mul_vec(x, y):
    t = 2.0 * _fast_cross(x[..., 1:], y)
    return y + x[..., 0][..., np.newaxis] * t + _fast_cross(x[..., 1:], t)


def mul_scalar(x, y):
    return slerp(eye_like(x[..., 0]), x, y)


def inv(x):
    return np.array([1, -1, -1, -1], dtype=np.float32) * x


def abs(x):
    return np.where((np.sum(x * np.array([1, 0, 0, 0], dtype=np.float32), axis=-1) > 0.0)[..., np.newaxis], x, -x)


def log(x, eps=1e-5):
    length = np.sqrt(np.sum(np.square(x[..., 1:]), axis=-1))[..., np.newaxis]
    halfangle = np.where(length < eps, np.ones_like(length), np.arctan2(length, x[..., 0:1]) / length)
    return halfangle * x[..., 1:]


def exp(x, eps=1e-5):
    halfangle = np.sqrt(np.sum(np.square(x), axis=-1))[..., np.newaxis]
    c = np.where(halfangle < eps, np.ones_like(halfangle), np.cos(halfangle))
    s = np.where(halfangle < eps, np.ones_like(halfangle), np.sinc(halfangle / np.pi))
    return np.concatenate([c, s * x], axis=-1)


def to_helical(x, eps=1e-5):
    return 2.0 * log(x, eps)


def from_helical(x, eps=1e-5):
    return exp(x / 2.0, eps)


def to_angle_axis(x, eps=1e-10):
    length = np.sqrt(np.sum(np.square(x[..., 1:]), axis=-1))
    angle = 2.0 * np.arctan2(length, x[..., 0])
    return angle, x[..., 1:] / (length + eps)


def from_angle_axis(angle, axis):
    c = np.cos(angle / 2.0)[..., np.newaxis]
    s = np.sin(angle / 2.0)[..., np.newaxis]
    return np.concatenate([c, s * axis], axis=-1)


def diff(x, y, world=True):
    diff = np.sum(x * y, axis=-1)[..., np.newaxis]
    flip = np.where(diff > 0.0, x, -x)
    return mul(flip, inv(y)) if world else mul(inv(y), flip)


def normalize(x, eps=0.0):
    return x / (np.sqrt(np.sum(x * x, axis=-1, keepdims=True)) + eps)


def between(x, y):
    return np.concatenate([
        np.sqrt(np.sum(x * x, axis=-1) * np.sum(y * y, axis=-1))[..., np.newaxis] +
        np.sum(x * y, axis=-1)[..., np.newaxis],
        _fast_cross(x, y)], axis=-1)


def slerp(x, y, a, eps=1e-10):
    l = np.sum(x * y, axis=-1)
    o = np.arccos(np.clip(l, -1.0, 1.0))
    a0 = np.sin((1.0 - a) * o) / (np.sin(o) + eps)
    a1 = np.sin((a) * o) / (np.sin(o) + eps)
    return a0[..., np.newaxis] * x + a1[..., np.newaxis] * y


def to_euler(x, order='zyx'):
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]

    if order == 'zyx':
        return np.concatenate([
            np.arctan2(2.0 * (x0 * x3 + x1 * x2), 1.0 - 2.0 * (x2 * x2 + x3 * x3)),
            np.arcsin(np.clip(2.0 * (x0 * x2 - x3 * x1), -1.0, 1.0)),
            np.arctan2(2.0 * (x0 * x1 + x2 * x3), 1.0 - 2.0 * (x1 * x1 + x2 * x2)),
        ], axis=-1)
    elif order == 'xzy':
        return np.concatenate([
            np.arctan2(2.0 * (x1 * x0 - x2 * x3), -x1 * x1 + x2 * x2 - x3 * x3 + x0 * x0),
            np.arctan2(2.0 * (x2 * x0 - x1 * x3), x1 * x1 - x2 * x2 - x3 * x3 + x0 * x0),
            np.arcsin(np.clip(2.0 * (x1 * x2 + x3 * x0), -1.0, 1.0))
        ], axis=-1)
    else:
        raise NotImplementedError('Cannot convert to ordering %s' % order)


def unroll(x):
    y = x.copy()
    for i in range(1, len(x)):
        d0 = np.sum(y[i] * y[i - 1], axis=-1)
        d1 = np.sum(-y[i] * y[i - 1], axis=-1)
        y[i][d0 < d1] = -y[i][d0 < d1]
    return y


def to_xform(x):
    qw, qx, qy, qz = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]

    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, wx = qx * x2, qy * y2, qw * x2
    xy, yz, wy = qx * y2, qy * z2, qw * y2
    xz, zz, wz = qx * z2, qz * z2, qw * z2

    return np.concatenate([
        np.concatenate([1.0 - (yy + zz), xy - wz, xz + wy], axis=-1)[..., np.newaxis, :],
        np.concatenate([xy + wz, 1.0 - (xx + zz), yz - wx], axis=-1)[..., np.newaxis, :],
        np.concatenate([xz - wy, yz + wx, 1.0 - (xx + yy)], axis=-1)[..., np.newaxis, :],
    ], axis=-2)


def from_euler(e, order='zyx'):
    axis = {'x': np.array([1, 0, 0], dtype=np.float32),
            'y': np.array([0, 1, 0], dtype=np.float32),
            'z': np.array([0, 0, 1], dtype=np.float32)}

    q0 = from_angle_axis(e[..., 0], axis[order[0]])
    q1 = from_angle_axis(e[..., 1], axis[order[1]])
    q2 = from_angle_axis(e[..., 2], axis[order[2]])

    return mul(q0, mul(q1, q2))


def from_xform(ts, eps=1e-10):
    qs = np.empty_like(ts[..., :1, 0].repeat(4, axis=-1))

    t = ts[..., 0, 0] + ts[..., 1, 1] + ts[..., 2, 2]

    s = 0.5 / np.sqrt(np.maximum(t + 1, eps))
    qs = np.where((t > 0)[..., np.newaxis].repeat(4, axis=-1), np.concatenate([
        (0.25 / s)[..., np.newaxis],
        (s * (ts[..., 2, 1] - ts[..., 1, 2]))[..., np.newaxis],
        (s * (ts[..., 0, 2] - ts[..., 2, 0]))[..., np.newaxis],
        (s * (ts[..., 1, 0] - ts[..., 0, 1]))[..., np.newaxis]
    ], axis=-1), qs)

    c0 = (ts[..., 0, 0] > ts[..., 1, 1]) & (ts[..., 0, 0] > ts[..., 2, 2])
    s0 = 2.0 * np.sqrt(np.maximum(1.0 + ts[..., 0, 0] - ts[..., 1, 1] - ts[..., 2, 2], eps))
    qs = np.where(((t <= 0) & c0)[..., np.newaxis].repeat(4, axis=-1), np.concatenate([
        ((ts[..., 2, 1] - ts[..., 1, 2]) / s0)[..., np.newaxis],
        (s0 * 0.25)[..., np.newaxis],
        ((ts[..., 0, 1] + ts[..., 1, 0]) / s0)[..., np.newaxis],
        ((ts[..., 0, 2] + ts[..., 2, 0]) / s0)[..., np.newaxis]
    ], axis=-1), qs)

    c1 = (~c0) & (ts[..., 1, 1] > ts[..., 2, 2])
    s1 = 2.0 * np.sqrt(np.maximum(1.0 + ts[..., 1, 1] - ts[..., 0, 0] - ts[..., 2, 2], eps))
    qs = np.where(((t <= 0) & c1)[..., np.newaxis].repeat(4, axis=-1), np.concatenate([
        ((ts[..., 0, 2] - ts[..., 2, 0]) / s1)[..., np.newaxis],
        ((ts[..., 0, 1] + ts[..., 1, 0]) / s1)[..., np.newaxis],
        (s1 * 0.25)[..., np.newaxis],
        ((ts[..., 1, 2] + ts[..., 2, 1]) / s1)[..., np.newaxis]
    ], axis=-1), qs)

    c2 = (~c0) & (~c1)
    s2 = 2.0 * np.sqrt(np.maximum(1.0 + ts[..., 2, 2] - ts[..., 0, 0] - ts[..., 1, 1], eps))
    qs = np.where(((t <= 0) & c2)[..., np.newaxis].repeat(4, axis=-1), np.concatenate([
        ((ts[..., 1, 0] - ts[..., 0, 1]) / s2)[..., np.newaxis],
        ((ts[..., 0, 2] + ts[..., 2, 0]) / s2)[..., np.newaxis],
        ((ts[..., 1, 2] + ts[..., 2, 1]) / s2)[..., np.newaxis],
        (s2 * 0.25)[..., np.newaxis]
    ], axis=-1), qs)

    return qs


def fk(lrot, lpos, parents):
    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(mul_vec(gr[parents[i]], lpos[..., i:i + 1, :]) + gp[parents[i]])
        gr.append(mul(gr[parents[i]], lrot[..., i:i + 1, :]))

    return np.concatenate(gr, axis=-2), np.concatenate(gp, axis=-2)


def fk_vel(lrot, lpos, lvrt, lvel, parents):
    gp, gr, gt, gv = [lpos[..., :1, :]], [lrot[..., :1, :]], [lvrt[..., :1, :]], [lvel[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(mul_vec(gr[parents[i]], lpos[..., i:i + 1, :]) + gp[parents[i]])
        gr.append(mul(gr[parents[i]], lrot[..., i:i + 1, :]))
        gt.append(gt[parents[i]] + mul_vec(gr[parents[i]], lvrt[..., i:i + 1, :]))
        gv.append(gv[parents[i]] + mul_vec(gr[parents[i]], lvel[..., i:i + 1, :]) +
                  _fast_cross(gt[parents[i]], mul_vec(gr[parents[i]], lpos[..., i:i + 1, :])))

    return np.concatenate(gr, axis=-2), np.concatenate(gp, axis=-2), np.concatenate(gt, axis=-2), np.concatenate(gv,
                                                                                                                 axis=-2)
    
def dot(a, b):
    return np.sum(a*b, axis=-1, keepdims=True)


def two_joint_ik(a, b, c, t, eps, a_gr, b_gr, a_lr, b_lr):

    b_a = b - a
    b_c = b - c
    t_a = t - a
    c_a = c - a
  
    lab = np.linalg.norm(b_a, ord=2, axis=-1, keepdims=True)
    lcb = np.linalg.norm(b_c, ord=2, axis=-1, keepdims=True)
    lat = np.linalg.norm(t_a, ord=2, axis=-1, keepdims=True)
    lca = np.linalg.norm(c_a, ord=2, axis=-1, keepdims=True)

    c_a = c_a / lca
    b_a = b_a / lab
    b_c = b_c / lcb
    t_a = t_a / lat

    ac_ab_0 = np.arccos(np.clip(dot(c_a, b_a), -1, 1))
    ba_bc_0 = np.arccos(np.clip(dot(b_a, b_c), -1, 1))
    ac_at_0 = np.arccos(np.clip(dot(c_a, t_a), -1, 1))

    lat = np.clip(lat, eps, lab + lcb - eps)

    ac_ab_1 = np.arccos(np.clip((lcb**2 - lab**2 - lat**2) / (-2*lab*lat), -1, 1))
    ba_bc_1 = np.arccos(np.clip((lat**2 - lab**2 - lcb**2) / (-2*lab*lcb), -1, 1))

    axis0 = normalize(np.cross(c_a, b_a))
    axis1 = normalize(np.cross(c_a, t_a))
    
    r0 = from_angle_axis((ac_ab_1 - ac_ab_0)[..., 0], mul_vec(inv(a_gr), axis0))
    r1 = from_angle_axis((ba_bc_1 - ba_bc_0)[..., 0], mul_vec(inv(b_gr), axis0))
    r2 = from_angle_axis(ac_at_0[..., 0], mul_vec(inv(a_gr), axis1))

    a_lr = mul(a_lr, mul(r0, r2))
    b_lr = mul(b_lr, r1)
    return a_lr, b_lr


def ik(lrot, lpos, parents, thigh, knee, foot, target):
    gp, gr = fk(lrot, lpos, parents)
    lrot[..., thigh, :], lrot[..., knee, :] = two_joint_ik(gp[..., thigh, :], gp[..., knee, :], gp[..., foot, :],
                                                           target, 1e-5,
                                                           gr[..., thigh, :], gr[..., knee, :],
                                                           lrot[..., thigh, :], lrot[..., knee, :])
    return lrot, lpos

        




