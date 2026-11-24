import datetime
import os
import shutil

import torch


def get_device(use_gpu=True):
    """CUDA if available, then Apple MPS, else CPU."""
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if use_gpu and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_useful_info(dest_path) -> None:
    """Snapshot the zeggs package source next to a training run."""
    dst = os.path.join(dest_path, "code")
    if not os.path.exists(dst):
        shutil.copytree(os.path.dirname(os.path.abspath(__file__)), dst,
                        ignore=shutil.ignore_patterns("__pycache__"))


def flatten_dict(dd, separator="_", prefix=""):
    return (
        {
            prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()
        }
        if isinstance(dd, dict)
        else {prefix: dd}
    )


def split_by_ratio(length, ratio):
    assert sum(ratio) == 1.0
    se = [0, 0]
    split = []
    for r in ratio:
        l = r * length
        s = int(se[-1])
        e = int(se[-1] + l)
        se = [s, e]
        split.append(se)
    split[-1][-1] = length
    return split


def percent_bar(ratio=1.0, width=30, empty=' ', done='#', parts=' -=>'):
    if ratio == 1.0:
        return done * width
    else:
        return (
                (done * width)[:int((100 * ratio) // (100 / width))] +
                (parts)[int(len(parts) * (((100 * ratio) / (100 / width)) % 1.0))] +
                (empty * width)[:max(width - int((100 * ratio) // (100 / width)) - 1, 0)])


def progress(ei, ii, bi, train_err, iter_num, start_time):
    percent = (float(bi) + 1) / (iter_num)
    curr_time = datetime.datetime.now()
    eta_time = start_time + (1.0 / (percent + 1e-10)) * (curr_time - start_time)

    return ("| %5i | %6i | [%s] %6.2f%% | % 8.4f | %s |" %
            (ei, ii, percent_bar(percent), 100 * percent, train_err, str(eta_time)[11:19]))
