""" S2G Training """
import datetime
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from anim import quat
from anim.tquat import *
from anim.txform import *
from dataset import SGDataset
from helpers import flatten_dict
from helpers import progress
from helpers import save_useful_info
from modules import Decoder
from modules import SpeechEncoder
from modules import StyleEncoder
from modules import compute_KL_div
from modules import normalize
from optimizers import RAdam
from utils import write_bvh


def train(
        models_dir,
        logs_dir,
        path_processed_data,
        path_data_definition,
        train_options,
        network_options,
):
    # ===============================================
    #    Getting/Setting Training/Network Configs
    # ===============================================
    np.random.seed(train_options["seed"])
    torch.manual_seed(train_options["seed"])
    torch.set_num_threads(train_options["thread_count"])
    use_gpu = train_options["use_gpu"] and torch.cuda.is_available()
    use_script = train_options["use_script"]

    if use_gpu:
        print("Using GPU!")
    else:
        print("Using CPU!")
    device = torch.device("cuda:0" if use_gpu else "cpu")
    window = train_options["window"]
    niterations = train_options["niterations"]
    batchsize = train_options["batchsize"]
    style_encoder_opts = network_options["style_encoder"]
    speech_encoder_opts = network_options["speech_encoder"]
    decoder_opts = network_options["decoder"]

    # ===============================================
    #                   Load Details
    # ===============================================
    with open(path_data_definition, "r") as f:
        details = json.load(f)

    nlabels = len(details["label_names"])
    bone_names = details["bone_names"]
    parents = torch.LongTensor(details["parents"])
    dt = details["dt"]

    # ===============================================
    #                   Load Data
    # ===============================================
    ds = SGDataset(
        path_data_definition,
        path_processed_data,
        window,
        style_encoding_type=train_options["style_encoding_type"],
        example_window_length=style_encoder_opts["example_length"],
    )
    # Workaround: The number of workers should be 0 so that the example length can be changed dynamically
    dl = DataLoader(ds, drop_last=True, batch_size=batchsize, shuffle=True, num_workers=0)

    dimensions = ds.get_shapes()

    (
        audio_input_mean,
        audio_input_std,
        anim_input_mean,
        anim_input_std,
        anim_output_mean,
        anim_output_std,
    ) = ds.get_means_stds(device)

    # ===============================================
    #             Load or Resume Networks
    # ===============================================
    style_encoding_type = train_options["style_encoding_type"]
    if style_encoding_type == "label":
        style_encoding_size = nlabels
    elif style_encoding_type == "example":
        style_encoding_size = style_encoder_opts["style_encoding_size"]

    path_network_speech_encoder_weights = models_dir / "speech_encoder.pt"
    path_network_decoder_weights = models_dir / "decoder.pt"
    path_network_style_encoder_weights = models_dir / "style_encoder.pt"
    path_checkpoints = models_dir / "checkpoints.pt"

    if (
            train_options["resume"]
            and os.path.exists(path_network_speech_encoder_weights)
            and os.path.exists(path_network_decoder_weights)
            and os.path.exists(path_checkpoints)
    ):
        network_speech_encoder = torch.load(path_network_speech_encoder_weights).to(device)
        network_decoder = torch.load(path_network_decoder_weights).to(device)
        network_style_encoder = torch.load(path_network_style_encoder_weights).to(device)

    else:
        network_speech_encoder = SpeechEncoder(
            dimensions["num_audio_features"],
            speech_encoder_opts["nhidden"],
            speech_encoder_opts["speech_encoding_size"],
        ).to(device)

        network_decoder = Decoder(
            pose_input_size=dimensions["pose_input_size"],
            pose_output_size=dimensions["pose_output_size"],
            speech_encoding_size=speech_encoder_opts["speech_encoding_size"],
            style_encoding_size=style_encoding_size,
            hidden_size=decoder_opts["nhidden"],
            num_rnn_layers=2,
        ).to(device)
        if style_encoding_type == "example":
            network_style_encoder = StyleEncoder(
                dimensions["pose_input_size"],
                style_encoder_opts["nhidden"],
                style_encoding_size,
                type=style_encoder_opts["type"],
                use_vae=style_encoder_opts["use_vae"],
            ).to(device)

    if use_script:
        network_speech_encoder_script = torch.jit.script(network_speech_encoder)
        network_decoder_script = torch.jit.script(network_decoder)
        if style_encoding_type == "example":
            network_style_encoder_script = torch.jit.script(network_style_encoder)
    else:
        network_speech_encoder_script = network_speech_encoder
        network_decoder_script = network_decoder
        if style_encoding_type == "example":
            network_style_encoder_script = network_style_encoder

    # ===============================================
    #                   Optimizer
    # ===============================================
    all_parameters = (
            list(network_speech_encoder.parameters())
            + list(network_decoder.parameters())
            + (list(network_style_encoder.parameters() if style_encoding_type == "example" else []))
    )
    optimizer = RAdam(all_parameters, lr=train_options["learning_rate"], eps=train_options["eps"])

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, train_options["learning_rate_decay"]
    )

    if train_options["resume"]:
        checkpoints = torch.load(path_checkpoints)
        iteration = checkpoints["iteration"]
        epoch = checkpoints["epoch"]
        loss = checkpoints["loss"]
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
    else:
        iteration = 0
        epoch = 0

    # ===============================================
    #             Setting Log Directories
    # ===============================================
    samples_dir = logs_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    if train_options["use_tensorboard"]:
        tb_dir = logs_dir / "tb"
        tb_dir.mkdir(exist_ok=True)
        writer = SummaryWriter(tb_dir, flush_secs=10)
        hparams = flatten_dict(network_options)
        hparams.update(flatten_dict(train_options))
        writer.add_hparams(hparams, {"No Metric": 0.0})

    # ===============================================
    #                 Begin Training
    # ===============================================
    while iteration < (1000 * niterations):

        start_time = datetime.datetime.now()

        for batch_index, batch in enumerate(dl):
            network_speech_encoder.train()
            network_decoder.train()
            if style_encoding_type == "example":
                network_style_encoder.train()
            (
                W_audio_features,
                W_root_pos,
                W_root_rot,
                W_root_vel,
                W_root_vrt,
                W_lpos,
                W_ltxy,
                W_lvel,
                W_lvrt,
                W_gaze_pos,
                WStyle,
            ) = batch

            W_audio_features = W_audio_features.to(device)
            W_root_pos = W_root_pos.to(device)
            W_root_rot = W_root_rot.to(device)
            W_root_vel = W_root_vel.to(device)
            W_root_vrt = W_root_vrt.to(device)
            W_lpos = W_lpos.to(device)
            W_ltxy = W_ltxy.to(device)
            W_lvel = W_lvel.to(device)
            W_lvrt = W_lvrt.to(device)
            W_gaze_pos = W_gaze_pos.to(device)
            WStyle = WStyle.to(device)

            # Dynamically changing example length for the next iteration
            ds.example_window_length = 2 * random.randint(style_encoder_opts["example_length"] // 2,
                                                          style_encoder_opts["example_length"])

            # Speech Encoder
            speech_encoding = network_speech_encoder_script(
                (W_audio_features - audio_input_mean) / audio_input_std
            )

            # Style Encoder
            mu, logvar = None, None
            if style_encoding_type == "example":
                WStyle = (WStyle - anim_input_mean) / anim_input_std
                style_encoding, mu, logvar = network_style_encoder_script(
                    WStyle.to(device=device)
                )
            else:
                style_encoding = WStyle

            # Gesture Generator
            (
                O_root_pos,
                O_root_rot,
                O_root_vel,
                O_root_vrt,
                O_lpos,
                O_ltxy,
                O_lvel,
                O_lvrt,
            ) = network_decoder_script(
                W_root_pos[:, 0],
                W_root_rot[:, 0],
                W_root_vel[:, 0],
                W_root_vrt[:, 0],
                W_lpos[:, 0],
                W_ltxy[:, 0],
                W_lvel[:, 0],
                W_lvrt[:, 0],
                W_gaze_pos,
                speech_encoding,
                style_encoding.unsqueeze(1).repeat((1, speech_encoding.shape[1], 1)),
                parents,
                anim_input_mean,
                anim_input_std,
                anim_output_mean,
                anim_output_std,
                dt,
            )

            # Compute Character/World Space
            W_lmat = xform_orthogonalize_from_xy(W_ltxy)
            O_lmat = xform_orthogonalize_from_xy(O_ltxy)

            ## Root Velocities to World Space
            O_root_vel_1_ = quat_mul_vec(O_root_rot[:, :-1], O_root_vel[:, 1:])
            O_root_vrt_1_ = quat_mul_vec(O_root_rot[:, :-1], O_root_vrt[:, 1:])
            O_root_vel_0 = quat_mul_vec(O_root_rot[:, 0:1], O_root_vel[:, 0:1])
            O_root_vrt_0 = quat_mul_vec(O_root_rot[:, 0:1], O_root_vrt[:, 0:1])
            O_root_vel = torch.cat((O_root_vel_0, O_root_vel_1_), dim=1)
            O_root_vrt = torch.cat((O_root_vrt_0, O_root_vrt_1_), dim=1)

            W_root_vel_1_ = quat_mul_vec(W_root_rot[:, :-1], W_root_vel[:, 1:])
            W_root_vrt_1_ = quat_mul_vec(W_root_rot[:, :-1], W_root_vrt[:, 1:])
            W_root_vel_0 = quat_mul_vec(W_root_rot[:, 0:1], W_root_vel[:, 0:1])
            W_root_vrt_0 = quat_mul_vec(W_root_rot[:, 0:1], W_root_vrt[:, 0:1])
            W_root_vel = torch.cat((W_root_vel_0, W_root_vel_1_), dim=1)
            W_root_vrt = torch.cat((W_root_vrt_0, W_root_vrt_1_), dim=1)

            ## Update First Joint
            O_lpos_0 = quat_mul_vec(O_root_rot, O_lpos[:, :, 0]) + O_root_pos
            O_lmat_0 = torch.matmul(quat_to_xform(O_root_rot), O_lmat[:, :, 0])
            O_lvel_0 = (
                    O_root_vel
                    + quat_mul_vec(O_root_rot, O_lvel[:, :, 0])
                    + torch.cross(O_root_vrt, quat_mul_vec(O_root_rot, O_lpos[:, :, 0]))
            )
            O_lvrt_0 = O_root_vrt + quat_mul_vec(O_root_rot, O_lvrt[:, :, 0])

            O_lpos = torch.cat((O_lpos_0.unsqueeze(2), O_lpos[:, :, 1:]), dim=2)
            O_lmat = torch.cat((O_lmat_0.unsqueeze(2), O_lmat[:, :, 1:]), dim=2)
            O_lvel = torch.cat((O_lvel_0.unsqueeze(2), O_lvel[:, :, 1:]), dim=2)
            O_lvrt = torch.cat((O_lvrt_0.unsqueeze(2), O_lvrt[:, :, 1:]), dim=2)

            W_lpos_0 = quat_mul_vec(W_root_rot, W_lpos[:, :, 0]) + W_root_pos
            W_lmat_0 = torch.matmul(quat_to_xform(W_root_rot), W_lmat[:, :, 0])
            W_lvel_0 = (
                    W_root_vel
                    + quat_mul_vec(W_root_rot, W_lvel[:, :, 0])
                    + torch.cross(W_root_vrt, quat_mul_vec(W_root_rot, W_lpos[:, :, 0]))
            )
            W_lvrt_0 = W_root_vrt + quat_mul_vec(W_root_rot, W_lvrt[:, :, 0])

            W_lpos = torch.cat((W_lpos_0.unsqueeze(2), W_lpos[:, :, 1:]), dim=2)
            W_lmat = torch.cat((W_lmat_0.unsqueeze(2), W_lmat[:, :, 1:]), dim=2)
            W_lvel = torch.cat((W_lvel_0.unsqueeze(2), W_lvel[:, :, 1:]), dim=2)
            W_lvrt = torch.cat((W_lvrt_0.unsqueeze(2), W_lvrt[:, :, 1:]), dim=2)

            # Fk to Character or World Space
            W_cmat, W_cpos, W_cvrt, W_cvel = xform_fk_vel(
                W_lmat, W_lpos, W_lvrt, W_lvel, parents
            )
            O_cmat, O_cpos, O_cvrt, O_cvel = xform_fk_vel(
                O_lmat, O_lpos, O_lvrt, O_lvel, parents
            )

            O_root_mat = quat_to_xform(O_root_rot)
            W_root_mat = quat_to_xform(W_root_rot)

            # Compute Gaze Dirs
            W_gaze_dir = quat_inv_mul_vec(W_root_rot, normalize(W_gaze_pos - W_root_pos))
            O_gaze_dir = quat_inv_mul_vec(O_root_rot, normalize(W_gaze_pos - O_root_pos))

            # Compute Losses
            loss_root_pos = torch.mean(torch.abs(0.1 * (O_root_pos - W_root_pos)))
            loss_root_rot = torch.mean(torch.abs(10.0 * (O_root_mat - W_root_mat)))
            loss_root_vel = torch.mean(torch.abs(0.1 * (O_root_vel - W_root_vel)))
            loss_root_vrt = torch.mean(torch.abs(5.0 * (O_root_vrt - W_root_vrt)))

            loss_lpos = torch.mean(torch.abs(15.0 * (O_lpos - W_lpos)))
            loss_lrot = torch.mean(torch.abs(15.0 * (O_ltxy - W_ltxy)))
            loss_lvel = torch.mean(torch.abs(10.0 * (O_lvel - W_lvel)))
            loss_lvrt = torch.mean(torch.abs(7.0 * (O_lvrt - W_lvrt)))

            loss_cpos = torch.mean(torch.abs(0.1 * (O_cpos - W_cpos)))
            loss_crot = torch.mean(torch.abs(3.0 * (O_cmat - W_cmat)))
            loss_cvel = torch.mean(torch.abs(0.06 * (O_cvel - W_cvel)))
            loss_cvrt = torch.mean(torch.abs(1.25 * (O_cvrt - W_cvrt)))

            loss_ldvl = torch.mean(
                torch.abs(
                    7.0
                    * (
                            (O_lpos[:, 1:] - O_lpos[:, :-1]) / dt
                            - (W_lpos[:, 1:] - W_lpos[:, :-1]) / dt
                    )
                )
            )

            loss_ldvt = torch.mean(
                torch.abs(
                    8.0
                    * (
                            (O_ltxy[:, 1:] - O_ltxy[:, :-1]) / dt
                            - (W_ltxy[:, 1:] - W_ltxy[:, :-1]) / dt
                    )
                )
            )

            loss_cdvl = torch.mean(
                torch.abs(
                    0.06
                    * (
                            (O_cpos[:, 1:] - O_cpos[:, :-1]) / dt
                            - (W_cpos[:, 1:] - W_cpos[:, :-1]) / dt
                    )
                )
            )

            loss_cdvt = torch.mean(
                torch.abs(
                    1.25
                    * (
                            (O_cmat[:, 1:] - O_cmat[:, :-1]) / dt
                            - (W_cmat[:, 1:] - W_cmat[:, :-1]) / dt
                    )
                )
            )

            loss_gaze = torch.mean(torch.abs(10.0 * (O_gaze_dir - W_gaze_dir)))

            loss_kl_div = 0.0
            if mu is not None and logvar is not None:
                kl_div, kl_div_weight = compute_KL_div(mu, logvar, iteration)
                loss_kl_div = kl_div_weight * torch.mean(kl_div)

            loss = (
                           +loss_root_pos
                           + loss_root_rot
                           + loss_root_vel
                           + loss_root_vrt
                           + loss_lpos
                           + loss_lrot
                           + loss_lvel
                           + loss_lvrt
                           + loss_cpos
                           + loss_crot
                           + loss_cvel
                           + loss_cvrt
                           + loss_ldvl
                           + loss_ldvt
                           + loss_cdvl
                           + loss_cdvt
                           + loss_gaze
                           + loss_kl_div
                   ) / 18.0

            # Backward
            loss.backward()
            optimizer.step()

            # Zero Gradients
            optimizer.zero_grad()

            losses = loss.detach().item()
            if (iteration + 1) % 1000 == 0:
                scheduler.step()

            # ===================================================
            #           Logging, Generating Samples
            # ===================================================
            if train_options["use_tensorboard"]:
                writer.add_scalar("losses/total_loss", loss, iteration)

                writer.add_scalars(
                    "losses/losses",
                    {
                        "loss_root_pos": loss_root_pos,
                        "loss_root_rot": loss_root_rot,
                        "loss_root_vel": loss_root_vel,
                        "loss_root_vrt": loss_root_vrt,
                        "loss_lpos": loss_lpos,
                        "loss_lrot": loss_lrot,
                        "loss_lvel": loss_lvel,
                        "loss_lvrt": loss_lvrt,
                        "loss_cpos": loss_cpos,
                        "loss_crot": loss_crot,
                        "loss_cvel": loss_cvel,
                        "loss_cvrt": loss_cvrt,
                        "loss_ldvl": loss_ldvl,
                        "loss_ldvt": loss_ldvt,
                        "loss_cdvl": loss_cdvl,
                        "loss_cdvt": loss_cdvt,
                        "loss_gaze": loss_gaze,
                        "loss_kl_div": loss_kl_div,
                    },
                    iteration,
                )

            if (iteration + 1) % 1 == 0:
                sys.stdout.write(
                    "\r"
                    + progress(
                        epoch,
                        iteration,
                        batch_index,
                        np.mean(losses),
                        (len(ds) // batchsize),
                        start_time,
                    )
                )
            if iteration % train_options["generate_samples_step"] == 0:
                sys.stdout.write(
                    "\r|                             Saving Networks...                                  |"
                )

                torch.save(network_speech_encoder, path_network_speech_encoder_weights)
                torch.save(network_decoder, path_network_decoder_weights)
                if style_encoding_type == "example":
                    torch.save(network_style_encoder, path_network_style_encoder_weights)
                torch.save({
                    'iteration': iteration,
                    "epoch": epoch,
                    'loss': loss,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, models_dir / "checkpoints.pt")

                current_models_dir = models_dir / str(iteration)
                current_models_dir.mkdir(exist_ok=True)

                path_network_speech_encoder_weights_current = current_models_dir / "speech_encoder.pt"
                path_network_decoder_weights_current = current_models_dir / "decoder.pt"
                path_network_style_encoder_weights_current = current_models_dir / "style_encoder.pt"

                torch.save(network_speech_encoder, path_network_speech_encoder_weights_current)
                torch.save(network_decoder, path_network_decoder_weights_current)
                if style_encoding_type == "example":
                    torch.save(network_style_encoder, path_network_style_encoder_weights_current)
                torch.save({
                    'iteration': iteration,
                    "epoch": epoch,
                    'loss': loss,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, current_models_dir / "checkpoints.pt")

                with torch.no_grad():
                    network_speech_encoder.eval()
                    network_decoder.eval()
                    if style_encoding_type == "example":
                        network_style_encoder.eval()
                    sys.stdout.write(
                        "\r|                           Generating Animation...                               |"
                    )

                    # Write training animation
                    for i in range(3):
                        (
                            S_audio_features,
                            S_root_pos,
                            S_root_rot,
                            S_root_vel,
                            S_root_vrt,
                            S_lpos,
                            S_ltxy,
                            S_lvel,
                            S_lvrt,
                            S_gaze_pos,
                            label,
                            se,
                            range_index,
                        ) = ds.get_sample("train", 30)

                        speech_encoding = network_speech_encoder_script(
                            (S_audio_features.to(device=device) - audio_input_mean)
                            / audio_input_std
                        )

                        if style_encoding_type == "example":
                            example = ds.get_example(se, se, ds.example_window_length)
                            example = (example.to(device=device) - anim_input_mean) / anim_input_std
                            style_encoding, _, _ = network_style_encoder_script(example[np.newaxis])
                        else:
                            style_encoding = np.zeros([nlabels])
                            style_encoding[label] = 1.0
                            style_encoding = torch.as_tensor(
                                style_encoding, dtype=torch.float32, device=device
                            )[np.newaxis]

                        (
                            V_root_pos,
                            V_root_rot,
                            _,
                            _,
                            V_lpos,
                            V_ltxy,
                            _,
                            _,
                        ) = network_decoder_script(
                            S_root_pos[:, 0].to(device=device),
                            S_root_rot[:, 0].to(device=device),
                            S_root_vel[:, 0].to(device=device),
                            S_root_vrt[:, 0].to(device=device),
                            S_lpos[:, 0].to(device=device),
                            S_ltxy[:, 0].to(device=device),
                            S_lvel[:, 0].to(device=device),
                            S_lvrt[:, 0].to(device=device),
                            S_gaze_pos.to(device=device),
                            speech_encoding,
                            style_encoding.unsqueeze(1).repeat((1, speech_encoding.shape[1], 1)),
                            parents,
                            anim_input_mean,
                            anim_input_std,
                            anim_output_mean,
                            anim_output_std,
                            dt,
                        )

                        S_lrot = quat.from_xform(xform_orthogonalize_from_xy(S_ltxy).cpu().numpy())
                        V_lrot = quat.from_xform(xform_orthogonalize_from_xy(V_ltxy).cpu().numpy())

                        try:
                            current_label = details["label_names"][label]
                            write_bvh(
                                str(
                                    samples_dir
                                    / (
                                        f"iteration_{iteration}_train_ground_{i}_{current_label}.bvh"
                                    )
                                ),
                                S_root_pos[0].cpu().numpy(),
                                S_root_rot[0].cpu().numpy(),
                                S_lpos[0].cpu().numpy(),
                                S_lrot[0],
                                parents=parents.cpu().numpy(),
                                names=bone_names,
                                order="zyx",
                                dt=dt,
                            )

                            write_bvh(
                                str(
                                    samples_dir
                                    / (
                                        f"iteration_{iteration}_train_predict_{i}_{current_label}.bvh"
                                    )
                                ),
                                V_root_pos[0].cpu().numpy(),
                                V_root_rot[0].cpu().numpy(),
                                V_lpos[0].cpu().numpy(),
                                V_lrot[0],
                                parents=parents.cpu().numpy(),
                                names=bone_names,
                                order="zyx",
                                dt=dt,
                            )

                        except (PermissionError, OSError) as e:
                            print(e)

                    # Write validation animation

                    for i in range(3):
                        (
                            S_audio_features,
                            S_root_pos,
                            S_root_rot,
                            S_root_vel,
                            S_root_vrt,
                            S_lpos,
                            S_ltxy,
                            S_lvel,
                            S_lvrt,
                            S_gaze_pos,
                            label,
                            se,
                            range_index,
                        ) = ds.get_sample("valid", 30)

                        speech_encoding = network_speech_encoder_script(
                            (S_audio_features.to(device=device) - audio_input_mean)
                            / audio_input_std
                        )

                        if style_encoding_type == "example":
                            example = ds.get_example(se, se, ds.example_window_length)
                            example = (example.to(device=device) - anim_input_mean) / anim_input_std
                            style_encoding, _, _ = network_style_encoder_script(example[np.newaxis])
                        else:
                            style_encoding = np.zeros([nlabels])
                            style_encoding[label] = 1.0
                            style_encoding = torch.as_tensor(
                                style_encoding, dtype=torch.float32, device=device
                            )[np.newaxis]

                        (
                            V_root_pos,
                            V_root_rot,
                            _,
                            _,
                            V_lpos,
                            V_ltxy,
                            _,
                            _,
                        ) = network_decoder_script(
                            S_root_pos[:, 0].to(device=device),
                            S_root_rot[:, 0].to(device=device),
                            S_root_vel[:, 0].to(device=device),
                            S_root_vrt[:, 0].to(device=device),
                            S_lpos[:, 0].to(device=device),
                            S_ltxy[:, 0].to(device=device),
                            S_lvel[:, 0].to(device=device),
                            S_lvrt[:, 0].to(device=device),
                            S_gaze_pos.to(device=device),
                            speech_encoding,
                            style_encoding.unsqueeze(1).repeat((1, speech_encoding.shape[1], 1)),
                            parents,
                            anim_input_mean,
                            anim_input_std,
                            anim_output_mean,
                            anim_output_std,
                            dt,
                        )

                        S_lrot = quat.from_xform(xform_orthogonalize_from_xy(S_ltxy).cpu().numpy())
                        V_lrot = quat.from_xform(xform_orthogonalize_from_xy(V_ltxy).cpu().numpy())

                        try:
                            current_label = details["label_names"][label]
                            write_bvh(
                                str(
                                    samples_dir
                                    / (
                                        f"iteration_{iteration}_valid_ground_{i}_{current_label}.bvh"
                                    )
                                ),
                                S_root_pos[0].cpu().numpy(),
                                S_root_rot[0].cpu().numpy(),
                                S_lpos[0].cpu().numpy(),
                                S_lrot[0],
                                parents=parents.cpu().numpy(),
                                names=bone_names,
                                order="zyx",
                                dt=dt,
                            )

                            write_bvh(
                                str(
                                    samples_dir
                                    / (
                                        f"iteration_{iteration}_valid_predict_{i}_{current_label}.bvh"
                                    )
                                ),
                                V_root_pos[0].cpu().numpy(),
                                V_root_rot[0].cpu().numpy(),
                                V_lpos[0].cpu().numpy(),
                                V_lrot[0],
                                parents=parents.cpu().numpy(),
                                names=bone_names,
                                order="zyx",
                                dt=dt,
                            )

                        except (PermissionError, OSError) as e:
                            print(e)

            iteration += 1
        sys.stdout.write("\n")

        epoch += 1
    print("Done!")


if __name__ == "__main__":

    # For debugging
    options = "../configs/configs_v2.json"
    with open(options, "r") as f:
        options = json.load(f)

    train_options = options["train_opt"]
    network_options = options["net_opt"]
    paths = options["paths"]

    base_path = Path(paths["base_path"])
    path_processed_data = base_path / paths["path_processed_data"] / "processed_data.npz"
    path_data_definition = base_path / paths["path_processed_data"] / "data_definition.json"

    # Output directory
    if paths["output_dir"] is None:
        output_dir = (base_path / "outputs") / datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        output_dir.mkdir(exist_ok=True, parents=True)
        paths["output_dir"] = str(output_dir)
    else:
        output_dir = Path(paths["output_dir"])

    # Path to models
    if paths["models_dir"] is None and not train_options["resume"]:
        models_dir = output_dir / "saved_models"
        models_dir.mkdir(exist_ok=True)
        paths["models_dir"] = str(models_dir)
    else:
        models_dir = Path(paths["models_dir"])

    # Log directory
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    options["paths"] = paths
    with open(output_dir / 'options.json', 'w') as fp:
        json.dump(options, fp, indent=4)

    save_useful_info(output_dir)

    train(
        models_dir=models_dir,
        logs_dir=logs_dir,
        path_processed_data=path_processed_data,
        path_data_definition=path_data_definition,
        train_options=train_options,
        network_options=network_options,
    )

    print("Done!")
