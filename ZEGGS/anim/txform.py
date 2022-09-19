import torch


def xform_transpose(xform):
    s = list(range(len(xform.shape)))
    s[-1], s[-2] = s[-2], s[-1]
    return xform.permute(*s)


def xform_fk_vel(lxform, lpos, lvrt, lvel, parents):
    gr, gp, gt, gv = [lxform[..., :1, :, :]], [lpos[..., :1, :]], [lvrt[..., :1, :]], [lvel[..., :1, :]]
    for i in range(1, len(parents)):
        p = parents[i]
        gp.append(gp[p] + torch.matmul(gr[p], lpos[..., i:i + 1, :][..., None])[..., 0])
        gr.append(torch.matmul(gr[p], lxform[..., i:i + 1, :, :]))
        gt.append(gt[p] + torch.matmul(gr[p], lvrt[..., i:i + 1, :][..., None])[..., 0])
        gv.append(gv[p] + torch.matmul(gr[p], lvel[..., i:i + 1, :][..., None])[..., 0] +
                  torch.cross(gt[p], torch.matmul(gr[p], lpos[..., i:i + 1, :][..., None])[..., 0], dim=-1))

    return torch.cat(gr, dim=-3), torch.cat(gp, dim=-2), torch.cat(gt, dim=-2), torch.cat(gv, dim=-2)


def xform_orthogonalize_from_xy(xy, eps=1e-10):
    xaxis = xy[..., 0:1, :]
    zaxis = torch.cross(xaxis, xy[..., 1:2, :])
    yaxis = torch.cross(zaxis, xaxis)

    output = torch.cat([
        xaxis / (torch.norm(xaxis, 2, dim=-1)[..., None] + eps),
        yaxis / (torch.norm(yaxis, 2, dim=-1)[..., None] + eps),
        zaxis / (torch.norm(zaxis, 2, dim=-1)[..., None] + eps)
    ], dim=-2)

    return xform_transpose(output)
