import argparse
import copy
import json
import os
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from rich.console import Console

from .anim import bvh
from .anim import quat
from .anim.txform import xform_orthogonalize_from_xy
from .audio.audio_files import read_wavfile
from .checkpoint import load_module
from .data_pipeline import preprocess_animation
from .data_pipeline import preprocess_audio
from .helpers import get_device
from .helpers import split_by_ratio
from .modules import example_feature_vector
from .postprocess import fix_foot_sliding, fix_heading_drift
from .utils import write_bvh

ANIM_FEATURES = ("root_pos root_rot root_vel root_vrt lpos lrot ltxy lvel lvrt "
                 "cpos crot ctxy cvel cvrt gaze_pos gaze_dir").split()


def _is_path(x):
    return isinstance(x, (str, os.PathLike))


def anim_features(anim_data, device):
    """preprocess_animation outputs as float32 tensors, keyed by name."""
    return {name: torch.as_tensor(value, dtype=torch.float32, device=device)
            for name, value in zip(ANIM_FEATURES, preprocess_animation(anim_data))}


def generate_gesture(
        audio_file,
        styles,
        network_path,
        data_path,
        results_path,
        style_encoding_type="example",
        blend_type="add",
        blend_ratio=[0.5, 0.5],
        file_name=None,
        first_pose=None,
        temperature=1.0,
        seed=1234,
        use_gpu=True,
        fix_feet=False,
        fix_heading=False,
):
    """Generate stylized gesture from raw audio and style example (ZEGGS)

    Args:
        audio_file: Path to audio file. If None the function does not generate gesture and only outputs the style embedding
        styles: What styles to use. Multiple styles are given for blending or stitching styles.
            Style Encoding Type == "example":
                This is a list of tuples S, where each tuple S provides info for one style.
                    - S[0] is the path to the bvh example or the style embedding vec to be used directly
                    - S[1] is a list or tuple of size two defining the start and end frame to be used.
                      For a style embedding vec, S[1] is its name.
            Style Encoding Type == "label":
                - List of style labels (names)
        network_path: Path to the networks
        data_path: Path to the data directory containing needed processing information
        results_path: Path to result directory
        style_encoding_type (str, optional): How to encode the style. Either "example" or "label". Defaults to "example".
        blend_type (str, optional): Blending type, stitch (transitioning) or add (mixing). Defaults to "add".
        blend_ratio (list, optional): The proportion of blending. If blend type is "stitch", this is the proportion of the length.
                                      of the output for this style. If the blend type is "add" this is the interpolation weight
                                      Defaults to [0.5, 0.5].
        file_name (optional): Output file name. If none the audio and example file names are used. Defaults to None.
        first_pose (optional): The first pose: a bvh path or an animation dictionary loaded from a bvh file.
                               If None, the first pose of the last style example is used (example-based stylization only).
        temperature (float, optional): VAE temperature. This adjusts the amount of stochasticity. Defaults to 1.0.
        seed (int, optional): Random seed. Defaults to 1234.
        use_gpu (bool, optional): Use CUDA or Apple MPS when available. Defaults to True.
        fix_feet (bool, optional): Pin planted feet to remove foot sliding (see zeggs.postprocess). Defaults to False.
        fix_heading (bool, optional): Keep the slow turning trend within 20 degrees of the start. Defaults to False.

    Returns:
        final_style_encoding: The final style embedding. If blend_type is "stitch", it is the style embedding for each frame.
                              If blend_type is "add", it is the interpolated style embedding vector
    """
    network_path, data_path = Path(network_path), Path(data_path)
    if results_path is not None:
        results_path = Path(results_path)
        results_path.mkdir(exist_ok=True, parents=True)
    assert (audio_file is None) == (results_path is None)

    torch.set_num_threads(1)
    device = get_device(use_gpu)

    # Data pipeline conf (We must use the same processing configuration as the one in training)
    with open(data_path / "data_pipeline_conf.json", "r") as f:
        data_pipeline_conf = DictConfig(json.load(f))

    # Animation static info (Skeleton, FPS, etc)
    with open(data_path / "data_definition.json", "r") as f:
        details = json.load(f)
    label_names = details["label_names"]
    bone_names = details["bone_names"]
    parents = torch.as_tensor(details["parents"], dtype=torch.long, device=device)
    dt = details["dt"]

    # Load Stats (Mean and Std of input/output)
    stat_data = np.load(data_path / "stats.npz")
    stats = {k: torch.as_tensor(stat_data[k], dtype=torch.float32, device=device) for k in (
        "audio_input_mean", "audio_input_std", "anim_input_mean", "anim_input_std",
        "anim_output_mean", "anim_output_std")}

    # Load Networks
    network_speech_encoder = load_module(network_path / "speech_encoder.pt", device)
    network_decoder = load_module(network_path / "decoder.pt", device)
    if style_encoding_type == "example":
        network_style_encoder = load_module(network_path / "style_encoder.pt", device)

    # Seed after building the networks: their initializers draw from the RNG.
    np.random.seed(seed)
    torch.manual_seed(seed)

    with torch.no_grad():
        # If audio is None we only output the style encodings
        if audio_file is not None:
            _, audio_data = read_wavfile(audio_file, rescale=True, desired_fs=16000, out_type="float32")
            n_frames = int(round(60.0 * (len(audio_data) / 16000)))
            audio_features = torch.as_tensor(
                preprocess_audio(audio_data, 60, n_frames, data_pipeline_conf.audio_conf,
                                 feature_type=data_pipeline_conf.audio_feature_type),
                device=device, dtype=torch.float32,
            )
            speech_encoding = network_speech_encoder(
                (audio_features[np.newaxis] - stats["audio_input_mean"]) / stats["audio_input_std"]
            )

        # Style Encoding
        style_encodings = []
        pose = None  # features of the last style example; its first frame seeds the decoder
        for style in styles:
            if style_encoding_type == "example":
                if _is_path(style[0]):
                    anim_name = Path(style[0]).stem
                    anim_data = bvh.load(style[0])

                    # Trimming if start/end frames are given
                    if style[1] is not None:
                        anim_data["rotations"] = anim_data["rotations"][style[1][0]: style[1][1]]
                        anim_data["positions"] = anim_data["positions"][style[1][0]: style[1][1]]
                    anim_fps = int(np.ceil(1 / anim_data["frametime"]))
                    assert anim_fps == 60

                    pose = anim_features(anim_data, device)
                    example = example_feature_vector(*(pose[k] for k in (
                        "root_vel", "root_vrt", "lpos", "ltxy", "lvel", "lvrt")))
                    example = (example - stats["anim_input_mean"]) / stats["anim_input_std"]
                    style_encoding, _, _ = network_style_encoder(example[np.newaxis], temperature)
                    style_encodings.append(style_encoding)

                elif isinstance(style[0], np.ndarray):
                    anim_name = style[1]
                    style_encodings.append(torch.as_tensor(style[0], dtype=torch.float32, device=device)[np.newaxis])
                else:
                    raise TypeError(f"Style must be a bvh path or an embedding array, got {type(style[0]).__name__}")
            elif style_encoding_type == "label":
                anim_name = style
                style_embedding = torch.zeros((1, len(label_names)), dtype=torch.float32, device=device)
                style_embedding[0, label_names.index(style)] = 1.0
                style_encodings.append(style_embedding)
                assert first_pose is not None, "Label-based generation needs a first_pose"
            else:
                raise ValueError("Unknown style encoding type")

        if blend_type == "stitch":
            if len(style_encodings) > 1:
                if audio_file is None:
                    final_style_encoding = style_encodings
                else:
                    assert len(styles) == len(blend_ratio)
                    se = split_by_ratio(n_frames, blend_ratio)
                    final_style_encoding = torch.cat([
                        style_encoding.unsqueeze(1).repeat((1, se[i][-1] - se[i][0], 1))
                        for i, style_encoding in enumerate(style_encodings)
                    ], dim=1)
            else:
                final_style_encoding = style_encodings[0]
        elif blend_type == "add":
            if len(style_encodings) > 1:
                assert len(style_encodings) == len(blend_ratio)
                final_style_encoding = torch.matmul(
                    torch.stack(style_encodings, dim=1).transpose(2, 1),
                    torch.tensor(blend_ratio, device=device),
                )
            else:
                final_style_encoding = style_encodings[0]
        else:
            raise ValueError(f"Unknown blend type {blend_type}")

        if audio_file is not None:
            if first_pose is not None:
                # deepcopy: preprocess_animation must not modify the caller's dictionary
                anim_data = bvh.load(first_pose) if _is_path(first_pose) else copy.deepcopy(first_pose)
                pose = anim_features(anim_data, device)
            assert pose is not None, "No first pose: pass first_pose or a bvh style example"

            if final_style_encoding.dim() == 2:
                final_style_encoding = final_style_encoding.unsqueeze(1).repeat((1, speech_encoding.shape[1], 1))
            first = {k: pose[k][0][np.newaxis] for k in (
                "root_pos", "root_rot", "root_vel", "root_vrt", "lpos", "ltxy", "lvel", "lvrt")}
            (
                V_root_pos,
                V_root_rot,
                V_root_vel,
                V_root_vrt,
                V_lpos,
                V_ltxy,
                V_lvel,
                V_lvrt,
            ) = network_decoder(
                first["root_pos"],
                first["root_rot"],
                first["root_vel"],
                first["root_vrt"],
                first["lpos"],
                first["ltxy"],
                first["lvel"],
                first["lvrt"],
                pose["gaze_pos"][0: 0 + 1].repeat_interleave(speech_encoding.shape[1], dim=0)[np.newaxis],
                speech_encoding,
                final_style_encoding,
                parents,
                stats["anim_input_mean"],
                stats["anim_input_std"],
                stats["anim_output_mean"],
                stats["anim_output_std"],
                dt,
            )

            V_lrot = quat.from_xform(xform_orthogonalize_from_xy(V_ltxy).detach().cpu().numpy())

            if file_name is None:
                file_name = f"audio_{Path(audio_file).stem}_label_{anim_name}"
            write_bvh(
                str(results_path / (file_name + ".bvh")),
                V_root_pos[0].detach().cpu().numpy(),
                V_root_rot[0].detach().cpu().numpy(),
                V_lpos[0].detach().cpu().numpy(),
                V_lrot[0],
                parents=parents.detach().cpu().numpy(),
                names=bone_names,
                order="zyx",
                dt=dt,
                start_position=np.array([0, 0, 0]),
                start_rotation=np.array([1, 0, 0, 0]),
            )
            if fix_feet or fix_heading:
                out_bvh = results_path / (file_name + ".bvh")
                anim = bvh.load(out_bvh)
                anim = fix_heading_drift(anim) if fix_heading else anim
                bvh.save(out_bvh, fix_foot_sliding(anim) if fix_feet else anim)
            copyfile(audio_file, str(results_path / (file_name + ".wav")))
    return final_style_encoding


def main():
    # CLI for generating gesture from one pair of audio and style files or multiple pairs through a csv file
    # For full functionality, please use the generate_gesture function

    console = Console()

    parser = argparse.ArgumentParser(prog="ZEGGS", description="Generate samples by ZEGGS model")
    parser.add_argument("-o", "--options", type=str, required=True,
                        help="Options filename (generated during training)")
    parser.add_argument('-p', '--results_path', type=str,
                        help="Results path. Default is the 'results' directory in the folder containing networks",
                        nargs="?", const=None, required=False)
    parser.add_argument('-se', '--style_encoding_type', type=str,
                        help="Style encoding type either 'example' or 'label'", default="example", required=False)

    # 1. Generating gesture from a single pair of audio and style files
    parser.add_argument('-s', '--style', type=str, help="Path to style example file (or label name)", required=False)
    parser.add_argument('-a', '--audio', type=str, help="Path to audio file", required=False)
    parser.add_argument('-n', '--file_name', type=str,
                        help="Output file name. If not given it will be automatically constructed", required=False)
    parser.add_argument('-fp', '--first_pose', type=str, help="First pose bvh file", default=None, required=False)
    parser.add_argument('-t', '--temperature', type=float,
                        help="VAE temperature. This adjusts the amount of stochasticity.", nargs="?", default=1.0,
                        required=False)
    parser.add_argument('-r', '--seed', type=int, help="Random seed", nargs="?", default=1234, required=False)
    parser.add_argument('-g', '--use_gpu', help="Use CUDA or Apple MPS (default is CPU)", action="store_true",
                        required=False)
    parser.add_argument('--fix_feet', action="store_true", help="Remove foot sliding by pinning planted feet")
    parser.add_argument('--fix_heading', action="store_true",
                        help="Keep the character from slowly turning away (slow trend within 20 degrees of the start)")
    parser.add_argument('-f', '--frames', type=int, help="Start and end frame of the style example to be used", nargs=2,
                        required=False)

    # 2. Generating gesture(s) from a csv file (some of the other arguments will be ignored)
    parser.add_argument('-c', '--csv', type=str,
                        help="CSV file containing information about pairs of audio/style and other parameters",
                        required=False)

    args = parser.parse_args()

    with open(args.options, "r") as f:
        options = json.load(f)
    paths = options["paths"]
    data_path = Path(paths["base_path"]) / paths["path_processed_data"]
    network_path = Path(paths["models_dir"])
    results_path = Path(args.results_path) if args.results_path else Path(paths["output_dir"]) / "results"
    style_encoding_type = args.style_encoding_type

    if args.csv is not None:
        console.print("Getting arguments from CSV file")
        df = pd.read_csv(args.csv)
        for index, row in df.iterrows():
            if not row["generate"]:
                continue

            with console.status(console.rule(f"Generating Gesture {index + 1}/{len(df)}")):
                console.print("Arguments:")
                console.print(row.to_string(index=True))
                base_path = Path(row["base_path"])
                frames = [int(x) for x in row["frames"].split(" ")] if isinstance(row["frames"], str) else None
                style = ([(base_path / row["style"], frames)] if style_encoding_type == "example"
                         else [row["style"]])
                generate_gesture(
                    audio_file=base_path / row["audio"],
                    styles=style,
                    network_path=network_path,
                    data_path=data_path,
                    results_path=results_path,
                    style_encoding_type=style_encoding_type,
                    file_name=row["file_name"],
                    first_pose=base_path / row["first_pose"],
                    temperature=row["temperature"],
                    seed=row["seed"],
                    use_gpu=row["use_gpu"],
                    fix_feet=args.fix_feet,
                    fix_heading=args.fix_heading,
                )
    else:
        with console.status(console.rule("Generating Gesture")):
            console.print("Arguments:")
            console.print(pd.DataFrame([vars(args)]).iloc[0].to_string(index=True))
            style = [(Path(args.style), args.frames)] if style_encoding_type == "example" else [args.style]
            generate_gesture(
                audio_file=Path(args.audio),
                styles=style,
                network_path=network_path,
                data_path=data_path,
                results_path=results_path,
                style_encoding_type=style_encoding_type,
                file_name=args.file_name,
                first_pose=args.first_pose,
                temperature=args.temperature,
                seed=args.seed,
                use_gpu=args.use_gpu,
                fix_feet=args.fix_feet,
                fix_heading=args.fix_heading,
            )


if __name__ == "__main__":
    main()
