import argparse
import json
import pathlib
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rich.console import Console

from anim import bvh, quat
from anim.txform import *
from audio.audio_files import read_wavfile
from data_pipeline import preprocess_animation, preprocess_audio
from helpers import split_by_ratio
from utils import write_bvh


def generate_gesture(
        audio_file,
        styles,
        network_path,
        data_path,
        results_path,
        blend_type="add",
        blend_ratio=[0.5, 0.5],
        file_name=None,
        first_pose=None,
        temperature=1.0,
        seed=1234,
        use_gpu=True,
        use_script=False,
):
    """Generate stylized gesture from raw audio and style example (ZEGGS)

    Args:
        audio_file ([type]): Path to audio file. If None the function does not generate geture and only outputs the style embedding
        styles ([type]): What styles to use. This is a list of tuples S, where each tuple S provides info for one style.
        Multiple styles are given for blending or stitching styles. Tuple S contains:
            - S[0] is the path to the bvh example or the style embedding vec to be used directly
            - S[1] is a list or tuple of size two defining the start and end frame to be used. None if style embedding is used directly
        network_path ([type]): Path to the networks
        data_path ([type]): Path to the data directory containing needed processing information
        results_path ([type]): Path to result directory
        blend_type (str, optional): Blending type, stitch (transitioning) or add (mixing). Defaults to "add".
        blend_ratio (list, optional): The proportion of blending. If blend type is "stitch", this is the proportion of the length. 
                                      of the output for this style. If the blend type is "add" this is the interpolation weight 
                                      Defaults to [0.5, 0.5].
        file_name ([type], optional): Output file name. If none the audio and example file names are used. Defaults to None.
        first_pose ([type], optional): The info required as the first pose. It can either be the path to the bvh file for using
                                       first pose or the animation dictionary extracted by loading a bvh file. 
                                       If None, the pose from the last example is used. Defaults to None.
        temperature (float, optional): VAE temprature. This adjusts the amount of stochasticity. Defaults to 1.0.
        seed (int, optional): Random seed. Defaults to 1234.
        use_gpu (bool, optional): Use gpu or cpu. Defaults to True.
        use_script (bool, optional): Use torch script. Defaults to False.

    Returns:
        final_style_encoding: The final style embedding. If blend_type is "stitch", it is the style embedding for each frame. 
                              If blend_type is "add", it is the interpolated style embedding vector
    """

    # Load details
    path_network_speech_encoder_weights = network_path / "speech_encoder.pt"
    path_network_decoder_weights = network_path / "decoder.pt"
    path_network_style_encoder_weights = network_path / "style_encoder.pt"
    path_stat_data = data_path / "stats.npz"
    path_data_definition = data_path / "data_definition.json"
    path_data_pipeline_conf = data_path / "data_pipeline_conf.json"
    if results_path is not None:
        results_path.mkdir(exist_ok=True)
    assert (audio_file is None) == (results_path is None)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    use_gpu = use_gpu and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")

    # Data pipeline conf (We must use the same processing configuration as the one in training)
    with open(path_data_pipeline_conf, "r") as f:
        data_pipeline_conf = json.load(f)
    data_pipeline_conf = DictConfig(data_pipeline_conf)

    # Animation static info (Skeleton, FPS, etc)
    with open(path_data_definition, "r") as f:
        details = json.load(f)

    njoints = len(details["bone_names"])
    nlabels = len(details["label_names"])
    bone_names = details["bone_names"]
    parents = torch.as_tensor(details["parents"], dtype=torch.long, device=device)
    dt = details["dt"]

    # Load Stats (Mean and Std of input/output)

    stat_data = np.load(path_stat_data)
    audio_input_mean = torch.as_tensor(
        stat_data["audio_input_mean"], dtype=torch.float32, device=device
    )
    audio_input_std = torch.as_tensor(
        stat_data["audio_input_std"], dtype=torch.float32, device=device
    )
    anim_input_mean = torch.as_tensor(
        stat_data["anim_input_mean"], dtype=torch.float32, device=device
    )
    anim_input_std = torch.as_tensor(
        stat_data["anim_input_std"], dtype=torch.float32, device=device
    )
    anim_output_mean = torch.as_tensor(
        stat_data["anim_output_mean"], dtype=torch.float32, device=device
    )
    anim_output_std = torch.as_tensor(
        stat_data["anim_output_std"], dtype=torch.float32, device=device
    )

    # Load Networks
    network_speech_encoder = torch.load(path_network_speech_encoder_weights).to(device)
    network_speech_encoder.eval()

    network_decoder = torch.load(path_network_decoder_weights).to(device)
    network_decoder.eval()

    network_style_encoder = torch.load(path_network_style_encoder_weights).to(device)
    network_style_encoder.eval()

    if use_script:
        network_speech_encoder_script = torch.jit.script(network_speech_encoder)
        network_decoder_script = torch.jit.script(network_decoder)
        network_style_encoder_script = torch.jit.script(network_style_encoder)
    else:
        network_speech_encoder_script = network_speech_encoder
        network_decoder_script = network_decoder
        network_style_encoder_script = network_style_encoder

    network_speech_encoder_script.eval()
    network_decoder_script.eval()
    network_style_encoder_script.eval()

    with torch.no_grad():
        # If audio is None we only output the style encodings
        if audio_file is not None:
            # Load Audio

            _, audio_data = read_wavfile(
                audio_file,
                rescale=True,
                desired_fs=16000,
                desired_nb_channels=None,
                out_type="float32",
                logger=None,
            )

            n_frames = int(round(60.0 * (len(audio_data) / 16000)))

            audio_features = torch.as_tensor(
                preprocess_audio(
                    audio_data,
                    60,
                    n_frames,
                    data_pipeline_conf.audio_conf,
                    feature_type=data_pipeline_conf.audio_feature_type,
                ),
                device=device,
                dtype=torch.float32,
            )
            speech_encoding = network_speech_encoder_script(
                (audio_features[np.newaxis] - audio_input_mean) / audio_input_std
            )

        # Style Encoding
        style_encodings = []

        for example in styles:
            if isinstance(example[0], pathlib.WindowsPath) or isinstance(example[0], pathlib.PosixPath):
                anim_name = Path(example[0]).stem
                anim_data = bvh.load(example[0])

                # Trimming if start/end frames are given
                if example[1] is not None:
                    anim_data["rotations"] = anim_data["rotations"][
                                             example[1][0]: example[1][1]
                                             ]
                    anim_data["positions"] = anim_data["positions"][
                                             example[1][0]: example[1][1]
                                             ]
                anim_fps = int(np.ceil(1 / anim_data["frametime"]))
                assert anim_fps == 60

                # Extracting features
                (
                    root_pos,
                    root_rot,
                    root_vel,
                    root_vrt,
                    lpos,
                    lrot,
                    ltxy,
                    lvel,
                    lvrt,
                    cpos,
                    crot,
                    ctxy,
                    cvel,
                    cvrt,
                    gaze_pos,
                    gaze_dir,
                ) = preprocess_animation(anim_data)

                # convert to tensor
                nframes = len(anim_data["rotations"])
                root_vel = torch.as_tensor(root_vel, dtype=torch.float32, device=device)
                root_vrt = torch.as_tensor(root_vrt, dtype=torch.float32, device=device)
                root_pos = torch.as_tensor(root_pos, dtype=torch.float32, device=device)
                root_rot = torch.as_tensor(root_rot, dtype=torch.float32, device=device)
                lpos = torch.as_tensor(lpos, dtype=torch.float32, device=device)
                ltxy = torch.as_tensor(ltxy, dtype=torch.float32, device=device)
                lvel = torch.as_tensor(lvel, dtype=torch.float32, device=device)
                lvrt = torch.as_tensor(lvrt, dtype=torch.float32, device=device)
                gaze_pos = torch.as_tensor(gaze_pos, dtype=torch.float32, device=device)

                S_root_vel = root_vel.reshape(nframes, -1)
                S_root_vrt = root_vrt.reshape(nframes, -1)
                S_lpos = lpos.reshape(nframes, -1)
                S_ltxy = ltxy.reshape(nframes, -1)
                S_lvel = lvel.reshape(nframes, -1)
                S_lvrt = lvrt.reshape(nframes, -1)
                example_feature_vec = torch.cat(
                    [
                        S_root_vel,
                        S_root_vrt,
                        S_lpos,
                        S_ltxy,
                        S_lvel,
                        S_lvrt,
                        torch.zeros_like(S_root_vel),
                    ],
                    dim=1,
                )
                example_feature_vec = (example_feature_vec - anim_input_mean) / anim_input_std

                style_encoding, _, _ = network_style_encoder_script(
                    example_feature_vec[np.newaxis], temperature
                )
                style_encodings.append(style_encoding)
            elif isinstance(example[0], np.ndarray):
                anim_name = example[1]
                style_embeddding = torch.as_tensor(
                    example[0], dtype=torch.float32, device=device
                )[np.newaxis]
                style_encodings.append(style_embeddding)
        if blend_type == "stitch":
            if len(style_encodings) > 1:
                if audio_file is None:
                    final_style_encoding = style_encodings
                else:
                    assert len(styles) == len(blend_ratio)
                    se = split_by_ratio(n_frames, blend_ratio)
                    V_root_pos = []
                    V_root_rot = []
                    V_lpos = []
                    V_ltxy = []
                    final_style_encoding = []
                    for i, style_encoding in enumerate(style_encodings):
                        final_style_encoding.append(
                            style_encoding.unsqueeze(1).repeat((1, se[i][-1] - se[i][0], 1))
                        )
                    final_style_encoding = torch.cat(final_style_encoding, dim=1)
            else:
                final_style_encoding = style_encodings[0]
        elif blend_type == "add":
            # style_encoding = torch.mean(torch.stack(style_encodings), dim=0)
            if len(style_encodings) > 1:
                assert len(style_encodings) == len(blend_ratio)
                final_style_encoding = torch.matmul(
                    torch.stack(style_encodings, dim=1).transpose(2, 1),
                    torch.tensor(blend_ratio, device=device),
                )
            else:
                final_style_encoding = style_encodings[0]

        if audio_file is not None:
            se = np.array_split(np.arange(n_frames), len(style_encodings))
            if first_pose is not None:
                if isinstance(first_pose, pathlib.WindowsPath) or isinstance(first_pose, pathlib.PosixPath):
                    anim_data = bvh.load(first_pose)
                elif isinstance(first_pose, dict):
                    anim_data = first_pose.copy()
                (
                    root_pos,
                    root_rot,
                    root_vel,
                    root_vrt,
                    lpos,
                    lrot,
                    ltxy,
                    lvel,
                    lvrt,
                    cpos,
                    crot,
                    ctxy,
                    cvel,
                    cvrt,
                    gaze_pos,
                    gaze_dir,
                ) = preprocess_animation(anim_data)

                root_vel = torch.as_tensor(root_vel, dtype=torch.float32, device=device)
                root_vrt = torch.as_tensor(root_vrt, dtype=torch.float32, device=device)
                root_pos = torch.as_tensor(root_pos, dtype=torch.float32, device=device)
                root_rot = torch.as_tensor(root_rot, dtype=torch.float32, device=device)
                lpos = torch.as_tensor(lpos, dtype=torch.float32, device=device)
                ltxy = torch.as_tensor(ltxy, dtype=torch.float32, device=device)
                lvel = torch.as_tensor(lvel, dtype=torch.float32, device=device)
                lvrt = torch.as_tensor(lvrt, dtype=torch.float32, device=device)
                gaze_pos = torch.as_tensor(gaze_pos, dtype=torch.float32, device=device)

            root_pos_0 = root_pos[0][np.newaxis]
            root_rot_0 = root_rot[0][np.newaxis]
            root_vel_0 = root_vel[0][np.newaxis]
            root_vrt_0 = root_vrt[0][np.newaxis]
            lpos_0 = lpos[0][np.newaxis]
            ltxy_0 = ltxy[0][np.newaxis]
            lvel_0 = lvel[0][np.newaxis]
            lvrt_0 = lvrt[0][np.newaxis]

            if final_style_encoding.dim() == 2:
                final_style_encoding = final_style_encoding.unsqueeze(1).repeat((1, speech_encoding.shape[1], 1))
            (
                V_root_pos,
                V_root_rot,
                V_root_vel,
                V_root_vrt,
                V_lpos,
                V_ltxy,
                V_lvel,
                V_lvrt,
            ) = network_decoder_script(
                root_pos_0,
                root_rot_0,
                root_vel_0,
                root_vrt_0,
                lpos_0,
                ltxy_0,
                lvel_0,
                lvrt_0,
                gaze_pos[0: 0 + 1].repeat_interleave(speech_encoding.shape[1], dim=0)[
                    np.newaxis
                ],
                speech_encoding,
                final_style_encoding,
                parents,
                anim_input_mean,
                anim_input_std,
                anim_output_mean,
                anim_output_std,
                dt,
            )

            V_lrot = quat.from_xform(xform_orthogonalize_from_xy(V_ltxy).detach().cpu().numpy())

            if file_name is None:
                file_name = f"audio_{audio_file.stem}_label_{anim_name}"
            try:
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
                copyfile(audio_file, str(results_path / (file_name + ".wav")))

            except (PermissionError, OSError) as e:
                print(e)
    return final_style_encoding


if __name__ == "__main__":

    # CLI for generating gesture from one pair of audio and style files or multiple pairs through a csv file
    # For full functionality, please use the generate_gesture function

    console = Console()

    # Setting parser
    parser = argparse.ArgumentParser(prog="ZEGGS", description="Generate samples by ZEGGS model")

    parser.add_argument(
        "-o",
        "--options",
        type=str,
        help="Options filename (generated during training)",
    )
    parser.add_argument('-p', '--results_path', type=str,
                        help="Results path. Default if 'results' directory in the folder containing networks",
                        nargs="?", const=None, required=False)

    # 1. Generating gesture from a single pair of audio and style files
    parser.add_argument('-s', '--style', type=str, help="Path to style example file", required=False)
    parser.add_argument('-a', '--audio', type=str, help="Path to audio file", required=False)
    parser.add_argument('-n', '--file_name', type=str,
                        help="Output file name. If not given it will be automatically constructed", required=False)
    parser.add_argument('-t', '--temperature', type=float,
                        help="VAE temprature. This adjusts the amount of stochasticity.", nargs="?", default=1.0,
                        required=False)
    parser.add_argument('-r', '--seed', type=int, help="Random seed", nargs="?", default=1234, required=False)
    parser.add_argument('-g', '--use_gpu', help="Use GPU (Default is using CPU)", action="store_true", required=False)
    parser.add_argument('-f', '--frames', type=int, help="Start and end frame of the style example to be used", nargs=2,
                        required=False)

    # 2. Generating gesture(s) from a csv file (some of the other arguments will be ignored)
    parser.add_argument('-c', '--csv', type=str,
                        help="CSV file containing information about pairs of audio/style and other parameters",
                        required=False)

    args = parser.parse_args()

    with open(args.options, "r") as f:
        options = json.load(f)

    train_options = options["train_opt"]
    network_options = options["net_opt"]
    paths = options["paths"]

    base_path = Path(paths["base_path"])
    data_path = base_path / paths["path_processed_data"]

    network_path = Path(paths["models_dir"])
    output_path = Path(paths["output_dir"])

    results_path = args.results_path
    if results_path is None:
        results_path = Path(output_path) / "results"

    if args.csv is not None:
        console.print("Getting arguments from CSV file")
        df = pd.read_csv(args.csv)
        for index, row in df.iterrows():
            if not row["generate"]:
                continue

            with console.status(console.rule(f"Generating Gesture {index + 1}/{len(df)}")):
                row["results_path"] = results_path
                row["options"] = args.options
                base_path = Path(row["base_path"])
                frames = [int(x) for x in row["frames"].split(" ")] if isinstance(row["frames"], str) else None

                console.print("Arguments:")
                console.print(row.to_string(index=True))
                generate_gesture(
                    audio_file=base_path / Path(row["audio"]),
                    styles=[(base_path / Path(row["style"]), frames)],
                    network_path=network_path,
                    data_path=data_path,
                    results_path=results_path,
                    file_name=row["file_name"],
                    temperature=row["temperature"],
                    seed=row["seed"],
                    use_gpu=row["use_gpu"]
                )
    else:
        with console.status(console.rule("Generating Gesture")):
            console.print("Arguments:")
            df = pd.DataFrame([vars(args)])
            console.print(df.iloc[0].to_string(index=True))
            file_name = args.file_name
            generate_gesture(
                audio_file=Path(args.audio),
                styles=[(Path(args.style), args.frames)],
                network_path=network_path,
                data_path=data_path,
                results_path=results_path,
                file_name=args.file_name,
                temperature=args.temperature,
                seed=args.seed,
                use_gpu=args.use_gpu
            )
