import json
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy import interpolate
from scipy.interpolate import griddata

from anim import bvh, quat
from audio.audio_files import read_wavfile, write_wavefile
from audio.spectrograms import extract_mel_spectrogram_for_tts

FILE_ROOT = os.path.dirname(os.path.realpath(__file__))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_logger = logging.getLogger(__name__)
_logger.propagate = False
warnings.simplefilter("ignore")


# ===============================================
#                   Audio
# ===============================================
def extract_energy(mel_spec):
    energy = np.linalg.norm(mel_spec, axis=0)
    return energy


def preprocess_audio(audio_data, anim_fs, anim_length, params, feature_type):
    if params.normalize_loudness:
        import pyloudnorm as pyln
        meter = pyln.Meter(params.sampling_rate)  # create BS.1770 meter
        loudness = meter.integrated_loudness(audio_data)
        # loudness normalize audio to -20 dB LUFS
        audio_data = pyln.normalize.loudness(audio_data, loudness, -20.0)

    resample_method = params.resample_method
    audio_feature = []
    # Extract MEL spectrogram
    mel_spec = extract_mel_spectrogram_for_tts(
        wav_signal=audio_data,
        fs=params.sampling_rate,
        n_fft=params.filter_length,
        step_size=params.hop_length,
        n_mels=params.n_mel_channels,
        mel_fmin=params.mel_fmin,
        mel_fmax=params.mel_fmax,
        min_amplitude=params.min_clipping,
        pre_emphasis=params.pre_emphasis,
        pre_emph_coeff=params.pre_emph_coeff,
        dynamic_range=None,
        real_amplitude=params.real_amplitude,
        centered=params.centered,
        normalize_mel_bins=params.normalize_mel_bins,
        normalize_range=params.normalize_range,
        logger=_logger,
    )[0].T
    mel_spec = 10 ** (mel_spec / 20)
    mel_spec = np.log(mel_spec)

    if "mel_spec" in feature_type:
        mel_spec_interp = interpolate.griddata(
            np.arange(len(mel_spec)),
            mel_spec,
            ((params.sampling_rate / params.hop_length) / anim_fs) * np.arange(anim_length),
            method=resample_method,
        ).astype(np.float32)
        audio_feature.append(mel_spec_interp)

    if "energy" in feature_type:
        energy = extract_energy(np.exp(mel_spec).T)
        f = interpolate.interp1d(np.arange(len(energy)), energy, kind=resample_method, fill_value="extrapolate")
        energy_interp = f(
            ((params.sampling_rate / params.hop_length) / anim_fs) * np.arange(anim_length)
        ).astype(np.float32)
        audio_feature.append(energy_interp[:, np.newaxis])

    audio_feature = np.concatenate(audio_feature, axis=1)

    return audio_feature


# ===============================================
#                   Animation
# ===============================================
def preprocess_animation(anim_data, conf=dict(), animation_path=None, info_df=None, i=0):
    nframes = len(anim_data["rotations"])
    njoints = len(anim_data["parents"])
    dt = anim_data["frametime"]

    lrot = quat.unroll(quat.from_euler(np.radians(anim_data["rotations"]), anim_data["order"]))

    lpos = anim_data["positions"]

    grot, gpos = quat.fk(lrot, lpos, anim_data["parents"])

    # Find root (Projected hips on the ground)
    root_pos = gpos[:, anim_data["names"].index("Spine2")] * np.array([1, 0, 1])
    # root_pos = signal.savgol_filter(root_pos, 31, 3, axis=0, mode="interp")

    # Root direction
    root_fwd = quat.mul_vec(grot[:, anim_data["names"].index("Hips")], np.array([[0, 0, 1]]))
    root_fwd[:, 1] = 0
    root_fwd = root_fwd / np.sqrt(np.sum(root_fwd * root_fwd, axis=-1))[..., np.newaxis]

    # root_fwd = signal.savgol_filter(root_fwd, 61, 3, axis=0, mode="interp")
    # root_fwd = root_fwd / np.sqrt(np.sum(root_fwd * root_fwd, axis=-1))[..., np.newaxis]

    # Root rotation
    root_rot = quat.normalize(
        quat.between(np.array([[0, 0, 1]]).repeat(len(root_fwd), axis=0), root_fwd)
    )

    # Find look at direction
    gaze_lookat = quat.mul_vec(grot[:, anim_data["names"].index("Head")], np.array([0, 0, 1]))
    gaze_lookat[:, 1] = 0
    gaze_lookat = gaze_lookat / np.sqrt(np.sum(np.square(gaze_lookat), axis=-1))[..., np.newaxis]

    # Find gaze position
    gaze_distance = 100  # Assume other actor is one meter away
    gaze_pos_all = root_pos + gaze_distance * gaze_lookat
    gaze_pos = np.median(gaze_pos_all, axis=0)
    gaze_pos = gaze_pos[np.newaxis].repeat(nframes, axis=0)

    # Visualize Gaze Pos
    if conf.get("visualize_gaze", False):
        import matplotlib.pyplot as plt

        plt.scatter(gaze_pos_all[:, 0], gaze_pos_all[:, 2], s=0.1, marker=".")
        plt.scatter(gaze_pos[0, 0], gaze_pos[0, 2])
        plt.scatter(root_pos[:, 0], root_pos[:, 2], s=0.1, marker=".")
        plt.quiver(root_pos[::60, 0], root_pos[::60, 2], root_fwd[::60, 0], root_fwd[::60, 2])
        plt.gca().set_aspect("equal")
        plt.show()

    # Compute local gaze dir
    gaze_dir = gaze_pos - root_pos
    # gaze_dir = gaze_dir / np.sqrt(np.sum(np.square(gaze_dir), axis=-1))[..., np.newaxis]
    gaze_dir = quat.mul_vec(quat.inv(root_rot), gaze_dir)

    # Make relative to root
    lrot[:, 0] = quat.mul(quat.inv(root_rot), lrot[:, 0])
    lpos[:, 0] = quat.mul_vec(quat.inv(root_rot), lpos[:, 0] - root_pos)

    # Local velocities
    lvel = np.zeros_like(lpos)
    lvel[1:] = (lpos[1:] - lpos[:-1]) / dt
    lvel[0] = lvel[1] - (lvel[3] - lvel[2])

    lvrt = np.zeros_like(lpos)
    lvrt[1:] = quat.to_helical(quat.abs(quat.mul(lrot[1:], quat.inv(lrot[:-1])))) / dt
    lvrt[0] = lvrt[1] - (lvrt[3] - lvrt[2])

    # Root velocities
    root_vrt = np.zeros_like(root_pos)
    root_vrt[1:] = quat.to_helical(quat.abs(quat.mul(root_rot[1:], quat.inv(root_rot[:-1])))) / dt
    root_vrt[0] = root_vrt[1] - (root_vrt[3] - root_vrt[2])
    root_vrt[1:] = quat.mul_vec(quat.inv(root_rot[:-1]), root_vrt[1:])
    root_vrt[0] = quat.mul_vec(quat.inv(root_rot[0]), root_vrt[0])

    root_vel = np.zeros_like(root_pos)
    root_vel[1:] = (root_pos[1:] - root_pos[:-1]) / dt
    root_vel[0] = root_vel[1] - (root_vel[3] - root_vel[2])
    root_vel[1:] = quat.mul_vec(quat.inv(root_rot[:-1]), root_vel[1:])
    root_vel[0] = quat.mul_vec(quat.inv(root_rot[0]), root_vel[0])

    # Compute character space
    crot, cpos, cvrt, cvel = quat.fk_vel(lrot, lpos, lvrt, lvel, anim_data["parents"])

    # Compute 2-axis transforms
    ltxy = np.zeros(dtype=np.float32, shape=[len(lrot), njoints, 2, 3])
    ltxy[..., 0, :] = quat.mul_vec(lrot, np.array([1.0, 0.0, 0.0]))
    ltxy[..., 1, :] = quat.mul_vec(lrot, np.array([0.0, 1.0, 0.0]))

    ctxy = np.zeros(dtype=np.float32, shape=[len(crot), njoints, 2, 3])
    ctxy[..., 0, :] = quat.mul_vec(crot, np.array([1.0, 0.0, 0.0]))
    ctxy[..., 1, :] = quat.mul_vec(crot, np.array([0.0, 1.0, 0.0]))

    if conf.get("save_normalized_animations", False):
        anim_data["positions"] = lpos
        anim_data["rotations"] = np.degrees(quat.to_euler(lrot, order=anim_data["order"]))

        normalized_animations_path = animation_path / "processed" / "normalized_animations"
        normalized_animations_path.mkdir(exist_ok=True)
        animation_norm_file = str(
            normalized_animations_path / info_df.iloc[i].anim_bvh).replace(
            ".bvh", "_norm.bvh"
        )

        bvh.save(animation_norm_file, anim_data)

        lpos_denorm = lpos.copy()
        lpos_denorm[:, 0] = quat.mul_vec(root_rot, lpos_denorm[:, 0]) + root_pos

        lrot_denorm = lrot.copy()
        lrot_denorm[:, 0] = quat.mul(root_rot, lrot_denorm[:, 0])

        anim_data["positions"] = lpos_denorm
        anim_data["rotations"] = np.degrees(quat.to_euler(lrot_denorm, order=anim_data["order"]))

        animation_denorm_file = str(
            animation_path / "processed" / "normalized_animations" / info_df.iloc[i].anim_bvh
        ).replace(".bvh", "_denorm.bvh")

        bvh.save(animation_denorm_file, anim_data)

    return (
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
    )


# ===============================================
#                   Pipeline
# ===============================================
def data_pipeline(conf):
    """Prepare Audio and Animation data for training

    Args:
        conf: config file

    Returns:
        processed_data, data_definition
    """
    from rich.progress import track
    from rich.console import Console
    from rich.table import Table
    console = Console(record=True)
    console.print("This may take a little bit of time ...")
    len_ratios = conf["len_ratios"]
    base_path = Path(conf["base_path"])
    processed_data_path = base_path / conf["processed_data_path"]
    processed_data_path.mkdir(exist_ok=True)
    info_filename = base_path / "info.csv"
    animation_path = base_path / "original"
    audio_path = base_path / "original"

    with open(str(processed_data_path / "data_pipeline_conf.json"), "w") as f:
        json.dump(conf, f, indent=4)
    conf = DictConfig(conf)

    info_df = pd.read_csv(info_filename)
    num_of_samples = len(info_df)
    audio_desired_fs = conf.audio_conf["sampling_rate"]

    X_audio_features = []

    Y_root_pos = []
    Y_root_rot = []
    Y_root_vrt = []
    Y_root_vel = []

    Y_lpos = []
    Y_lrot = []
    Y_ltxy = []
    Y_lvel = []
    Y_lvrt = []

    Y_gaze_pos = []
    Y_gaze_dir = []

    current_start_frame = 0
    ranges_train = []
    ranges_valid = []
    ranges_train_labels = []
    ranges_valid_labels = []

    for i in track(range(num_of_samples), description="Processing...", complete_style="magenta"):

        animation_file = str(animation_path / info_df.iloc[i].anim_bvh)
        audio_file = audio_path / info_df.iloc[i].audio_filename

        # Load Animation #
        original_anim_data = bvh.load(animation_file)
        anim_fps = int(np.ceil(1 / original_anim_data["frametime"]))
        assert anim_fps == 60

        # Load Audio #
        audio_sr, original_audio_data = read_wavfile(
            audio_file,
            rescale=True,
            desired_fs=audio_desired_fs,
            desired_nb_channels=None,
            out_type="float32",
            logger=_logger,
        )

        # Silence Audio #
        speacker_timing_df = pd.read_csv(audio_file.with_suffix(".csv"))

        # Mark regions that don't need silencing
        mask = np.zeros_like(original_audio_data)
        for ind, row in speacker_timing_df.iterrows():

            if "R" in row["#"]:
                start_time = [int(num) for num in row["Start"].replace(".", ":").rsplit(":")]
                end_time = [int(num) for num in row["End"].replace(".", ":").rsplit(":")]

                start_time = (
                        start_time[0] * 60 * audio_desired_fs
                        + start_time[1] * audio_desired_fs
                        + int(start_time[2] * (audio_desired_fs / 1000))
                )

                end_time = (
                        end_time[0] * 60 * audio_desired_fs
                        + end_time[1] * audio_desired_fs
                        + int(end_time[2] * (audio_desired_fs / 1000))
                )

                mask[start_time:end_time] = 1.0

        # Silence unmarked regions
        original_audio_data = original_audio_data * mask

        # Sync & Trim #
        # Get mark-ups
        audio_start_time = info_df.iloc[i].audio_start_time
        audio_start_time = [int(num) for num in audio_start_time.rsplit(":")]
        anim_start_time = info_df.iloc[i].anim_start_time
        anim_start_time = [int(num) for num in anim_start_time.rsplit(":")]
        acting_start_time = info_df.iloc[i].acting_start_time
        acting_start_time = [int(num) for num in acting_start_time.rsplit(":")]
        acting_end_time = info_df.iloc[i].acting_end_time
        acting_end_time = [int(num) for num in acting_end_time.rsplit(":")]

        # Compute Timings (This is assuming that audio timing is given in 30fps)
        audio_start_time_in_thirds = (
                audio_start_time[0] * 216000
                + audio_start_time[1] * 3600
                + audio_start_time[2] * 60
                + audio_start_time[3] * 2
        )

        anim_start_time_in_thirds = (
                anim_start_time[0] * 216000
                + anim_start_time[1] * 3600
                + anim_start_time[2] * 60
                + anim_start_time[3] * 1
        )

        acting_start_time_in_thirds = (
                acting_start_time[0] * 216000
                + acting_start_time[1] * 3600
                + acting_start_time[2] * 60
                + acting_start_time[3] * 1
        )

        acting_end_time_in_thirds = (
                acting_end_time[0] * 216000
                + acting_end_time[1] * 3600
                + acting_end_time[2] * 60
                + acting_end_time[3] * 1
        )

        acting_start_in_audio_ref = int(
            np.round(
                (acting_start_time_in_thirds - audio_start_time_in_thirds) * (audio_sr / 60)
            )
        )

        acting_end_in_audio_ref = int(
            np.round((acting_end_time_in_thirds - audio_start_time_in_thirds) * (audio_sr / 60))
        )

        acting_start_in_anim_ref = int(
            np.round(
                (acting_start_time_in_thirds - anim_start_time_in_thirds) * (anim_fps / 60)
            )
        )

        acting_end_in_anim_ref = int(
            np.round((acting_end_time_in_thirds - anim_start_time_in_thirds) * (anim_fps / 60))
        )

        if (
                acting_start_in_audio_ref < 0
                or acting_start_in_anim_ref < 0
                or acting_end_in_audio_ref < 0
                or acting_end_in_anim_ref < 0
        ):
            raise ValueError("The timings are incorrect!")

        # Trim to equal length
        original_audio_data = original_audio_data[acting_start_in_audio_ref:acting_end_in_audio_ref]

        original_anim_data["rotations"] = original_anim_data["rotations"][
                                          acting_start_in_anim_ref:acting_end_in_anim_ref
                                          ]

        original_anim_data["positions"] = original_anim_data["positions"][
                                          acting_start_in_anim_ref:acting_end_in_anim_ref
                                          ]
        for len_ratio in len_ratios:
            anim_data = original_anim_data.copy()
            audio_data = original_audio_data.copy()
            if len_ratio != 1.0:
                n_anim_frames = len(original_anim_data["rotations"])
                nbones = anim_data["positions"].shape[1]
                original_times = np.linspace(0, n_anim_frames - 1, n_anim_frames)
                sample_times = np.linspace(0, n_anim_frames - 1, int(len_ratio * (n_anim_frames)))
                anim_data["positions"] = griddata(original_times, anim_data["positions"].reshape([n_anim_frames, -1]),
                                                  sample_times, method='cubic').reshape([len(sample_times), nbones, 3])

                rotations = quat.unroll(quat.from_euler(np.radians(anim_data['rotations']), order=anim_data['order']))
                rotations = griddata(original_times, rotations.reshape([n_anim_frames, -1]), sample_times,
                                     method='cubic').reshape([len(sample_times), nbones, 4])
                rotations = quat.normalize(rotations)
                anim_data["rotations"] = np.degrees(quat.to_euler(rotations, order=anim_data["order"]))

                n_audio_frames = len(audio_data)
                original_times = np.linspace(0, n_audio_frames - 1, n_audio_frames)
                sample_times = np.linspace(0, n_audio_frames - 1, int(len_ratio * (n_audio_frames)))
                audio_data = griddata(original_times, audio_data, sample_times, method='cubic')
                # assert len(audio_data) / audio_sr == len(anim_data["rotations"]) / anim_fps

            # Saving Trimmed Files
            folder = "valid" if info_df.iloc[i].validation else "train"
            trimmed_filename = info_df.iloc[i].anim_bvh.split(".")[0]
            trimmed_filename = trimmed_filename + "_x_" + str(len_ratio).replace(".", "_")

            if conf["save_trimmed_audio"]:
                target_path = processed_data_path / "trimmed" / folder
                target_path.mkdir(exist_ok=True, parents=True)
                write_wavefile(target_path / (trimmed_filename + ".wav"), audio_data, audio_sr)

            if conf["save_trimmed_animation"]:
                target_path = processed_data_path / "trimmed" / folder
                target_path.mkdir(exist_ok=True, parents=True)

                # Centering the character. Comment if you want the original global position and orientation
                output = anim_data.copy()
                lrot = quat.from_euler(np.radians(output["rotations"]), output["order"])
                offset_pos = output["positions"][0:1, 0:1].copy() * np.array([1, 0, 1])
                offset_rot = lrot[0:1, 0:1].copy() * np.array([1, 0, 1, 0])

                root_pos = quat.mul_vec(quat.inv(offset_rot), output["positions"][:, 0:1] - offset_pos)
                output["positions"][:, 0:1] = quat.mul_vec(quat.inv(offset_rot),
                                                           output["positions"][:, 0:1] - offset_pos)
                output["rotations"][:, 0:1] = np.degrees(
                    quat.to_euler(quat.mul(quat.inv(offset_rot), lrot[:, 0:1]), order=output["order"]))

                bvh.save(target_path / (trimmed_filename + ".bvh"), anim_data)

            # Extracting Audio Features #
            audio_features = preprocess_audio(
                audio_data,
                anim_fps,
                len(anim_data["rotations"]),
                conf.audio_conf,
                feature_type=conf.audio_feature_type,
            )

            # Check if the lengths are correct and no NaNs
            assert len(audio_features) == len(anim_data["rotations"])
            assert not np.any(np.isnan(audio_features))

            if conf["visualize_spectrogram"]:
                import matplotlib.pyplot as plt

                plt.imshow(audio_features.T, interpolation="nearest")
                plt.show()

            # Extracting Animation Features
            nframes = len(anim_data["rotations"])
            dt = anim_data["frametime"]
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
            ) = preprocess_animation(anim_data, conf, animation_path, info_df, i)

            # Appending Data
            X_audio_features.append(audio_features)

            Y_root_pos.append(root_pos)
            Y_root_rot.append(root_rot)
            Y_root_vel.append(root_vel)
            Y_root_vrt.append(root_vrt)
            Y_lpos.append(lpos)
            Y_lrot.append(lrot)
            Y_ltxy.append(ltxy)
            Y_lvel.append(lvel)
            Y_lvrt.append(lvrt)

            Y_gaze_pos.append(gaze_pos)
            Y_gaze_dir.append(gaze_dir)

            # Append to Ranges
            current_end_frame = nframes + current_start_frame

            if info_df.iloc[i].validation:
                ranges_valid.append([current_start_frame, current_end_frame])
                ranges_valid_labels.append(info_df.iloc[i].style)
            else:
                ranges_train.append([current_start_frame, current_end_frame])
                ranges_train_labels.append(info_df.iloc[i].style)

            current_start_frame = current_end_frame

    # Processing Labels
    ranges_train = np.array(ranges_train, dtype=np.int32)
    ranges_valid = np.array(ranges_valid, dtype=np.int32)

    label_names = list(set(ranges_train_labels + ranges_valid_labels))

    ranges_train_labels = np.array(
        [label_names.index(label) for label in ranges_train_labels], dtype=np.int32
    )
    ranges_valid_labels = np.array(
        [label_names.index(label) for label in ranges_valid_labels], dtype=np.int32
    )

    # Concatenating Data
    X_audio_features = np.concatenate(X_audio_features, axis=0).astype(np.float32)

    Y_root_pos = np.concatenate(Y_root_pos, axis=0).astype(np.float32)
    Y_root_rot = np.concatenate(Y_root_rot, axis=0).astype(np.float32)
    Y_root_vel = np.concatenate(Y_root_vel, axis=0).astype(np.float32)
    Y_root_vrt = np.concatenate(Y_root_vrt, axis=0).astype(np.float32)

    Y_lpos = np.concatenate(Y_lpos, axis=0).astype(np.float32)
    Y_lrot = np.concatenate(Y_lrot, axis=0).astype(np.float32)
    Y_ltxy = np.concatenate(Y_ltxy, axis=0).astype(np.float32)
    Y_lvel = np.concatenate(Y_lvel, axis=0).astype(np.float32)
    Y_lvrt = np.concatenate(Y_lvrt, axis=0).astype(np.float32)

    Y_gaze_pos = np.concatenate(Y_gaze_pos, axis=0).astype(np.float32)
    Y_gaze_dir = np.concatenate(Y_gaze_dir, axis=0).astype(np.float32)

    # Compute Means & Stds
    # Filter out start and end frames
    ranges_mask = np.zeros(len(X_audio_features), dtype=bool)
    for s, e in ranges_train:
        ranges_mask[s + 2: e - 2] = True

    # Compute Means
    Y_root_vel_mean = Y_root_vel[ranges_mask].mean(axis=0)
    Y_root_vrt_mean = Y_root_vrt[ranges_mask].mean(axis=0)

    Y_lpos_mean = Y_lpos[ranges_mask].mean(axis=0)
    Y_ltxy_mean = Y_ltxy[ranges_mask].mean(axis=0)
    Y_lvel_mean = Y_lvel[ranges_mask].mean(axis=0)
    Y_lvrt_mean = Y_lvrt[ranges_mask].mean(axis=0)

    Y_gaze_dir_mean = Y_gaze_dir[ranges_mask].mean(axis=0)

    audio_input_mean = X_audio_features[ranges_mask].mean(axis=0)

    anim_input_mean = np.hstack(
        [
            Y_root_vel_mean.ravel(),
            Y_root_vrt_mean.ravel(),
            Y_lpos_mean.ravel(),
            Y_ltxy_mean.ravel(),
            Y_lvel_mean.ravel(),
            Y_lvrt_mean.ravel(),
            Y_gaze_dir_mean.ravel(),
        ]
    )

    # Compute Stds
    Y_root_vel_std = Y_root_vel[ranges_mask].std() + 1e-10
    Y_root_vrt_std = Y_root_vrt[ranges_mask].std() + 1e-10

    Y_lpos_std = Y_lpos[ranges_mask].std() + 1e-10
    Y_ltxy_std = Y_ltxy[ranges_mask].std() + 1e-10
    Y_lvel_std = Y_lvel[ranges_mask].std() + 1e-10
    Y_lvrt_std = Y_lvrt[ranges_mask].std() + 1e-10

    Y_gaze_dir_std = Y_gaze_dir[ranges_mask].std() + 1e-10

    audio_input_std = X_audio_features[ranges_mask].std() + 1e-10

    anim_input_std = np.hstack(
        [
            Y_root_vel_std.repeat(len(Y_root_vel_mean.ravel())),
            Y_root_vrt_std.repeat(len(Y_root_vrt_mean.ravel())),
            Y_lpos_std.repeat(len(Y_lpos_mean.ravel())),
            Y_ltxy_std.repeat(len(Y_ltxy_mean.ravel())),
            Y_lvel_std.repeat(len(Y_lvel_mean.ravel())),
            Y_lvrt_std.repeat(len(Y_lvrt_mean.ravel())),
            Y_gaze_dir_std.repeat(len(Y_gaze_dir_mean.ravel())),
        ]
    )

    # Compute Output Means
    anim_output_mean = np.hstack(
        [
            Y_root_vel_mean.ravel(),
            Y_root_vrt_mean.ravel(),
            Y_lpos_mean.ravel(),
            Y_ltxy_mean.ravel(),
            Y_lvel_mean.ravel(),
            Y_lvrt_mean.ravel(),
        ]
    )

    # Compute Output Stds
    Y_root_vel_out_std = Y_root_vel[ranges_mask].std(axis=0)
    Y_root_vrt_out_std = Y_root_vrt[ranges_mask].std(axis=0)

    Y_lpos_out_std = Y_lpos[ranges_mask].std(axis=0)
    Y_ltxy_out_std = Y_ltxy[ranges_mask].std(axis=0)
    Y_lvel_out_std = Y_lvel[ranges_mask].std(axis=0)
    Y_lvrt_out_std = Y_lvrt[ranges_mask].std(axis=0)

    anim_output_std = np.hstack(
        [
            Y_root_vel_out_std.ravel(),
            Y_root_vrt_out_std.ravel(),
            Y_lpos_out_std.ravel(),
            Y_ltxy_out_std.ravel(),
            Y_lvel_out_std.ravel(),
            Y_lvrt_out_std.ravel(),
        ]
    )

    processed_data = dict(
        X_audio_features=X_audio_features,
        Y_root_pos=Y_root_pos,
        Y_root_rot=Y_root_rot,
        Y_root_vel=Y_root_vel,
        Y_root_vrt=Y_root_vrt,
        Y_lpos=Y_lpos,
        Y_ltxy=Y_ltxy,
        Y_lvel=Y_lvel,
        Y_lvrt=Y_lvrt,
        Y_gaze_pos=Y_gaze_pos,
        ranges_train=ranges_train,
        ranges_valid=ranges_valid,
        ranges_train_labels=ranges_train_labels,
        ranges_valid_labels=ranges_valid_labels,
        audio_input_mean=audio_input_mean,
        audio_input_std=audio_input_std,
        anim_input_mean=anim_input_mean,
        anim_input_std=anim_input_std,
        anim_output_mean=anim_output_mean,
        anim_output_std=anim_output_std,
    )

    stats = dict(
        ranges_train=ranges_train,
        ranges_valid=ranges_valid,
        ranges_train_labels=ranges_train_labels,
        ranges_valid_labels=ranges_valid_labels,
        audio_input_mean=audio_input_mean,
        audio_input_std=audio_input_std,
        anim_input_mean=anim_input_mean,
        anim_input_std=anim_input_std,
        anim_output_mean=anim_output_mean,
        anim_output_std=anim_output_std,
    )

    data_definition = dict(
        dt=dt,
        label_names=label_names,
        parents=anim_data["parents"].tolist(),
        bone_names=anim_data["names"],
    )

    # Save Data
    if conf["save_final_data"]:
        np.savez(processed_data_path / "processed_data.npz", **processed_data)

        np.savez(processed_data_path / "stats.npz", **stats)

        with open(str(processed_data_path / "data_definition.json"), "w") as f:
            json.dump(data_definition, f, indent=4)

    # Data Stats:
    nlabels = len(label_names)
    df = pd.DataFrame()
    df["Dataset"] = ["Train", "Validation", "Total"]
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    table = Table(title="Data Info", show_lines=True, row_styles=["magenta"])
    table.add_column("Dataset")
    data_len = 0
    for i in range(nlabels):
        ind_mask = ranges_train_labels == i
        ranges = ranges_train[ind_mask]
        num_train_frames = (
                np.sum(ranges[:, 1] - ranges[:, 0]) / 2
        )  # It is divided by two as we have mirrored versions too
        ind_mask = ranges_valid_labels == i
        ranges = ranges_valid[ind_mask]
        num_valid_frames = np.sum(ranges[:, 1] - ranges[:, 0]) / 2
        total = num_train_frames + num_valid_frames
        df[label_names[i]] = [
            f"{num_train_frames} frames - {num_train_frames / 60:.1f} secs",
            f"{num_valid_frames} frames - {num_valid_frames / 60:.1f} secs",
            f"{total} frames - {total / 60:.1f} secs",
        ]
        table.add_column(label_names[i])
        data_len += total

    for i in range(3):
        table.add_row(*list(df.iloc[i]))
    console.print(table)
    console.print(f"Total length of dataset is {data_len} frames - {data_len / 60:.1f} seconds")
    console_print_file = processed_data_path / "data_info.html"
    console.print(dict(conf))
    console.save_html(str(console_print_file))

    return processed_data, data_definition


if __name__ == "__main__":
    config_file = "../configs/data_pipeline_conf_v1.json"
    with open(config_file, "r") as f:
        conf = json.load(f)

    data_pipeline(conf)
