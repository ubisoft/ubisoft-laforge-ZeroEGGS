"""Network checkpoints stored as {class, config, state_dict}, loadable with weights_only=True."""
import sys
from pathlib import Path

import torch

from . import modules

_CLASSES = {c.__name__: c for c in (modules.SpeechEncoder, modules.StyleEncoder, modules.Decoder)}


def save_module(module, config, path):
    torch.save({"class": type(module).__name__, "config": dict(config), "state_dict": module.state_dict()}, path)


def load_module(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    module = _CLASSES[ckpt["class"]](**ckpt["config"])
    module.load_state_dict(ckpt["state_dict"])
    return module.to(device).eval()


def _legacy_config(net, speech_encoding_size):
    """Constructor arguments of a network pickled by the original code, read from its weights."""
    sd = net.state_dict()
    if isinstance(net, modules.SpeechEncoder):
        w0, w1 = sd["layer0.weight"], sd["layer1.weight"]
        return dict(input_size=w0.shape[1], hidden_size=w0.shape[0], output_size=w1.shape[0])
    if isinstance(net, modules.StyleEncoder):
        w0 = sd["encoder.convs.0.conv.weight"]
        kind = "attn" if isinstance(net.encoder, modules.StyleEncoderAttn) else "gru"
        return dict(input_size=w0.shape[1], hidden_size=w0.shape[0], style_embedding_size=net.style_embedding_size,
                    type=kind, use_vae=net.use_vae)
    if isinstance(net, modules.Decoder):
        film = isinstance(net.recurrent_decoder, modules.RecurrentDecoderFiLM)
        pose_output_size = sd["recurrent_decoder.layer3.weight" if film else "recurrent_decoder.layer2.weight"].shape[0]
        pose_input_size = pose_output_size + 3  # the input also carries the 3D local gaze direction
        return dict(
            pose_input_size=pose_input_size,
            pose_output_size=pose_output_size,
            speech_encoding_size=speech_encoding_size,
            style_encoding_size=sd["cell_state_encoder.layer0.weight"].shape[1] - pose_input_size,
            hidden_size=sd["cell_state_encoder.layer1.weight"].shape[0],
            num_rnn_layers=net.cell_state_encoder.num_rnn_layers,
            rnn_cond="film" if film else "normal",
        )
    raise TypeError(f"Unknown network type {type(net).__name__}")


def convert_legacy_models(models_dir):
    """Rewrite the pickled networks in `models_dir` (original ZEGGS format) as state_dict checkpoints."""
    # The original pickles refer to top-level `modules` and `anim` packages.
    from . import anim
    sys.modules.setdefault("modules", modules)
    sys.modules.setdefault("anim", anim)
    models_dir = Path(models_dir)
    names = ["speech_encoder", "decoder", "style_encoder"]
    nets = {n: torch.load(models_dir / f"{n}.pt", map_location="cpu", weights_only=False)
            for n in names if (models_dir / f"{n}.pt").exists()}
    speech_size = _legacy_config(nets["speech_encoder"], None)["output_size"]
    for name, net in nets.items():
        config = _legacy_config(net, speech_size)
        rebuilt = _CLASSES[type(net).__name__](**config)
        rebuilt.load_state_dict(net.state_dict())
        save_module(rebuilt, config, models_dir / f"{name}.pt")
        print(f"converted {models_dir / name}.pt: {config}")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        convert_legacy_models(d)
