import torch
import math
from typing import Tuple

@torch.jit.script
def quat_mul(x, y):
    
    x0, x1, x2, x3 = x[...,0:1], x[...,1:2], x[...,2:3], x[...,3:4]
    y0, y1, y2, y3 = y[...,0:1], y[...,1:2], y[...,2:3], y[...,3:4]

    return torch.cat([
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,      
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,   
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,    
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0], dim=-1)
    
@torch.jit.script
def quat_mul_vec(x, y):
    t = 2.0 * torch.cross(x[...,1:], y, dim=-1)
    return y + x[...,0:1] * t + torch.cross(x[...,1:], t, dim=-1)

@torch.jit.script
def quat_inv(x):
    return torch.tensor([1,-1,-1,-1], dtype=torch.float32, device=x.device) * x

@torch.jit.script
def quat_inv_mul(x, y):
    return quat_mul(quat_inv(x), y)
    
@torch.jit.script
def quat_inv_mul_vec(x, y):
    return quat_mul_vec(quat_inv(x), y)

@torch.jit.script
def quat_abs(x):
    return torch.where(x[...,0:1] > 0.0, x, -x)

@torch.jit.script
def quat_diff(x, y, world : bool = True):
    diff = torch.sum(x * y, dim=-1, keepdim=True)
    flip = torch.where(diff > 0.0, x, -x)
    return quat_mul(flip, quat_inv(y)) if world else quat_mul(quat_inv(y), flip)

@torch.jit.script
def quat_diff_linear(x, y):
    d = quat_abs(quat_mul(x, quat_inv(y)))
    return 2.0 * d[...,1:]
    
@torch.jit.script
def quat_normalize(x, eps : float = 1e-5):
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)

@torch.jit.script
def quat_to_xform(x):

    qw, qx, qy, qz = x[...,0:1], x[...,1:2], x[...,2:3], x[...,3:4]
    
    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, wx = qx * x2, qy * y2, qw * x2
    xy, yz, wy = qx * y2, qy * z2, qw * y2
    xz, zz, wz = qx * z2, qz * z2, qw * z2
    
    return torch.cat([
        torch.cat([1.0 - (yy + zz), xy - wz, xz + wy], dim=-1)[...,None,:],
        torch.cat([xy + wz, 1.0 - (xx + zz), yz - wx], dim=-1)[...,None,:],
        torch.cat([xz - wy, yz + wx, 1.0 - (xx + yy)], dim=-1)[...,None,:],
    ], dim=-2)
    
@torch.jit.script
def quat_to_xy(x):

    qw, qx, qy, qz = x[...,0:1], x[...,1:2], x[...,2:3], x[...,3:4]
    
    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, wx = qx * x2, qy * y2, qw * x2
    xy, yz, wy = qx * y2, qy * z2, qw * y2
    xz, zz, wz = qx * z2, qz * z2, qw * z2
    
    return torch.cat([
        torch.cat([1.0 - (yy + zz), xy - wz], dim=-1)[...,None,:],
        torch.cat([xy + wz, 1.0 - (xx + zz)], dim=-1)[...,None,:],
        torch.cat([xz - wy, yz + wx        ], dim=-1)[...,None,:],
    ], dim=-2)
    
@torch.jit.script
def quat_log(x, eps : float = 1e-5):
    length = torch.norm(x[...,1:], dim=-1, keepdim=True)
    return torch.where(
        length < eps, 
        x[...,1:], 
        (torch.atan2(length, x[...,0:1]) / length) * x[...,1:])
    
@torch.jit.script
def quat_exp(x, eps : float = 1e-5):
    halfangle = torch.norm(x, dim=-1, keepdim=True)
    return torch.where(halfangle < eps, quat_normalize(
        torch.cat([torch.ones_like(halfangle), x], dim=-1)),
        torch.cat([torch.cos(halfangle),       x * torch.sinc(halfangle / math.pi)], dim=-1))
    
@torch.jit.script
def quat_to_helical(x, eps : float = 1e-5):
    return 2.0 * quat_log(x, eps)
    
@torch.jit.script
def quat_from_helical(x, eps : float = 1e-5):
    return quat_exp(x / 2.0, eps)
    
@torch.jit.script
def quat_from_helical_approx(v):
    return quat_normalize(torch.cat([
        torch.ones_like(v[...,:1]), v / 2.0], dim=-1))
        
@torch.jit.script
def quat_to_helical_approx(v):
    return 2.0 * v[...,:1]
    
@torch.jit.script
def quat_from_angle_axis(angle, axis):
    c = torch.cos(angle / 2.0)[...,None]
    s = torch.sin(angle / 2.0)[...,None]
    return torch.cat([c, s * axis], dim=-1)
    
@torch.jit.script
def quat_to_angle_axis(x, eps : float = 1e-5):
    length = torch.norm(x[...,1:], dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(length, x[...,0])
    return angle, x[...,1:] / (length + eps)
    
@torch.jit.script
def quat_ik_rot(grot, parents):
    lr = [grot[...,:1,:]]
    for i in range(1, len(parents)):
        p = parents[i]
        lr.append(quat_mul(quat_inv(grot[...,p:p+1,:]), grot[...,i:i+1,:]))
    return  torch.cat(lr, dim=-2)
    
@torch.jit.script
def quat_fk(lrot, lpos, parents):
    
    gp, gr = [lpos[...,:1,:]], [lrot[...,:1,:]]
    for i in range(1, len(parents)):
        gp.append(quat_mul_vec(gr[parents[i]], lpos[...,i:i+1,:]) + gp[parents[i]])
        gr.append(quat_mul    (gr[parents[i]], lrot[...,i:i+1,:]))
        
    return torch.cat(gr, dim=-2), torch.cat(gp, dim=-2)

@torch.jit.script
def quat_fk_vel(lrot, lpos, lvrt, lvel, parents):
    
    gp, gr, gt, gv = [lpos[...,:1,:]], [lrot[...,:1,:]], [lvrt[...,:1,:]], [lvel[...,:1,:]]
    for i in range(1, len(parents)):
        gp.append(quat_mul_vec(gr[parents[i]], lpos[...,i:i+1,:]) + gp[parents[i]])
        gr.append(quat_mul    (gr[parents[i]], lrot[...,i:i+1,:]))
        gt.append(gt[parents[i]] + quat_mul_vec(gr[parents[i]], lvrt[...,i:i+1,:]))
        gv.append(gv[parents[i]] + quat_mul_vec(gr[parents[i]], lvel[...,i:i+1,:]) + 
            torch.cross(gt[parents[i]], quat_mul_vec(gr[parents[i]], lpos[...,i:i+1,:]), dim=-1))
        
    return torch.cat(gr, dim=-2), torch.cat(gp, dim=-2), torch.cat(gt, dim=-2), torch.cat(gv, dim=-2)

@torch.jit.script
def quat_fk_root_position(gpos, grot, lpos, parents):
    gp = [gpos[...,:1,:]]
    for i in range(1, len(parents)):
        gp.append(quat_mul_vec(grot[...,parents[i]:parents[i]+1,:], lpos[...,i:i+1,:]) + gp[parents[i]])
    return torch.cat(gp, dim=-2)
    
@torch.jit.script
def quat_character_to_local(grot, parents):
    lr = [grot[...,:1,:]]
    for i in range(1, len(parents)):
        lr.append(quat_mul( quat_inv(grot[...,parents[i]:parents[i]+1,:]), grot[...,i:i+1,:]))
    return torch.cat(lr, dim=-2)


