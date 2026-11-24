import numpy as np
import torch
from scipy.io import wavfile

from zeggs.anim import bvh, quat, tquat
from zeggs.audio.audio_files import read_wavfile
from zeggs.checkpoint import load_module, save_module
from zeggs.modules import Decoder, SpeechEncoder, StyleEncoder

rng = np.random.default_rng(0)


def random_quats(n):
    return quat.normalize(rng.normal(size=(n, 4)))


def test_euler_round_trip():
    q = quat.abs(random_quats(100))
    for order in ("zyx", "xzy"):
        back = quat.abs(quat.from_euler(quat.to_euler(q, order), order))
        np.testing.assert_allclose(back, q, atol=1e-6)


def test_xform_round_trip():
    q = quat.abs(random_quats(100))
    np.testing.assert_allclose(quat.abs(quat.from_xform(quat.to_xform(q))), q, atol=1e-6)


def test_torch_and_numpy_quats_agree():
    q, v = random_quats(50), rng.normal(size=(50, 3))
    np.testing.assert_allclose(
        tquat.quat_mul_vec(torch.tensor(q), torch.tensor(v)).numpy(), quat.mul_vec(q, v), atol=1e-10)


def test_bvh_round_trip(tmp_path):
    names, parents = ["Hips", "Spine", "Head"], np.array([-1, 0, 1])
    positions = rng.normal(size=(5, 3, 3))
    rotations = rng.uniform(-80, 80, size=(5, 3, 3))
    data = dict(rotations=rotations, positions=positions, offsets=positions[0], parents=parents,
                names=names, order="zyx", frametime=1 / 60)
    bvh.save(tmp_path / "a.bvh", data, translations=True)
    back = bvh.load(tmp_path / "a.bvh")
    assert back["names"] == names and back["order"] == "zyx"
    np.testing.assert_array_equal(back["parents"], parents)
    np.testing.assert_allclose(back["positions"], positions, atol=1e-5)
    np.testing.assert_allclose(back["rotations"], rotations, atol=1e-5)


def test_read_wavfile_resamples_and_downmixes(tmp_path):
    t = np.arange(48000) / 48000
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    wavfile.write(tmp_path / "s.wav", 48000, (np.stack([tone, tone], 1) * 32767).astype(np.int16))
    fs, x = read_wavfile(tmp_path / "s.wav", rescale=True, desired_fs=16000)
    assert fs == 16000 and x.shape == (16000,) and x.dtype == np.float32
    assert np.argmax(np.abs(np.fft.rfft(x))) == 440  # 1 s of audio: bin index == Hz


def test_checkpoint_round_trip(tmp_path):
    nets = [
        (SpeechEncoder, dict(input_size=5, hidden_size=4, output_size=3)),
        (StyleEncoder, dict(input_size=7, hidden_size=8, style_embedding_size=4, type="attn", use_vae=True)),
        (Decoder, dict(pose_input_size=24, pose_output_size=21, speech_encoding_size=3, style_encoding_size=4,
                       hidden_size=8, num_rnn_layers=2, rnn_cond="film")),
    ]
    for cls, config in nets:
        net = cls(**config).eval()
        save_module(net, config, tmp_path / "net.pt")
        loaded = load_module(tmp_path / "net.pt")
        assert type(loaded) is cls and not loaded.training
        for key, value in net.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key], value)


def test_style_encoder_mean_pools_over_time():
    torch.manual_seed(0)
    net = StyleEncoder(input_size=7, hidden_size=8, style_embedding_size=4, type="attn", use_vae=False).eval()
    x = torch.randn(3, 20, 7)
    with torch.no_grad():
        batched, _, _ = net(x)
        single = torch.cat([net(x[i:i + 1])[0] for i in range(3)])
    torch.testing.assert_close(batched, single, atol=1e-5, rtol=1e-5)
