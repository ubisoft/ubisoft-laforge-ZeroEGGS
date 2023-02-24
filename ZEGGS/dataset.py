import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class SGDataset(Dataset):
    def __init__(
            self,
            path_data_definition,
            path_processed_data,
            window,
            style_encoding_type,
            example_window_length,
    ):
        """PyTorch Dataset Instance

        Args:
            path_data_definition : Path to data_definition file
            path_processed_data : Path to processed_data npz file
            window : Length of the input-output slice
            style_encoding_type : "label" or "example"
            example_window_length : Length of example window
        """

        with open(path_data_definition, "r") as f:
            details = json.load(f)
        self.details = details
        self.njoints = len(details["bone_names"])
        self.nlabels = len(details["label_names"])
        self.label_names = details["label_names"]
        self.bone_names = details["bone_names"]
        self.parents = torch.LongTensor(details["parents"])
        self.dt = details["dt"]
        self.window = window
        self.style_encoding_type = style_encoding_type
        self.example_window_length = example_window_length

        # Load Data
        processed_data = np.load(path_processed_data)

        self.ranges_train = processed_data["ranges_train"]
        self.ranges_valid = processed_data["ranges_valid"]
        self.ranges_train_labels = processed_data["ranges_train_labels"]
        self.ranges_valid_labels = processed_data["ranges_valid_labels"]

        self.X_audio_features = torch.as_tensor(
            processed_data["X_audio_features"], dtype=torch.float32
        )
        self.Y_root_pos = torch.as_tensor(processed_data["Y_root_pos"], dtype=torch.float32)
        self.Y_root_rot = torch.as_tensor(processed_data["Y_root_rot"], dtype=torch.float32)
        self.Y_root_vel = torch.as_tensor(processed_data["Y_root_vel"], dtype=torch.float32)
        self.Y_root_vrt = torch.as_tensor(processed_data["Y_root_vrt"], dtype=torch.float32)
        self.Y_lpos = torch.as_tensor(processed_data["Y_lpos"], dtype=torch.float32)
        self.Y_ltxy = torch.as_tensor(processed_data["Y_ltxy"], dtype=torch.float32)
        self.Y_lvel = torch.as_tensor(processed_data["Y_lvel"], dtype=torch.float32)
        self.Y_lvrt = torch.as_tensor(processed_data["Y_lvrt"], dtype=torch.float32)
        self.Y_gaze_pos = torch.as_tensor(processed_data["Y_gaze_pos"], dtype=torch.float32)

        self.audio_input_mean = torch.as_tensor(
            processed_data["audio_input_mean"], dtype=torch.float32
        )
        self.audio_input_std = torch.as_tensor(
            processed_data["audio_input_std"], dtype=torch.float32
        )
        self.anim_input_mean = torch.as_tensor(
            processed_data["anim_input_mean"], dtype=torch.float32
        )
        self.anim_input_std = torch.as_tensor(processed_data["anim_input_std"], dtype=torch.float32)
        self.anim_output_mean = torch.as_tensor(
            processed_data["anim_output_mean"], dtype=torch.float32
        )
        self.anim_output_std = torch.as_tensor(
            processed_data["anim_output_std"], dtype=torch.float32
        )

        # Build Windows
        R = []
        L = []
        S = []
        for sample_number, ((range_start, range_end), range_label) in enumerate(
                zip(self.ranges_train, self.ranges_train_labels)
        ):

            one_hot_label = np.zeros(self.nlabels, dtype=np.float32)
            one_hot_label[range_label] = 1.0

            for ri in range(range_start, range_end - window):
                R.append(np.arange(ri, ri + window))
                L.append(one_hot_label)
                S.append(sample_number)

        self.R = torch.as_tensor(np.array(R), dtype=torch.long)
        self.L = torch.as_tensor(np.array(L), dtype=torch.float32)
        self.S = torch.as_tensor(S, dtype=torch.short)
        # self.get_stats()

    @property
    def example_window_length(self):
        return self._example_window_length

    @example_window_length.setter
    def example_window_length(self, a):
        self._example_window_length = a

    def __len__(self):
        return len(self.R)

    def __getitem__(self, index):
        # Extract Windows
        Rwindow = self.R[index]
        Rwindow = Rwindow.contiguous()

        # Extract Labels
        Rlabel = self.L[index]

        # Get Corresponding Ranges for Style Encoding
        RInd = self.S[index]
        sample_range = self.ranges_train[RInd]

        # Extract Audio
        W_audio_features = self.X_audio_features[Rwindow]

        # Extract Animation
        W_root_pos = self.Y_root_pos[Rwindow]
        W_root_rot = self.Y_root_rot[Rwindow]
        W_root_vel = self.Y_root_vel[Rwindow]
        W_root_vrt = self.Y_root_vrt[Rwindow]
        W_lpos = self.Y_lpos[Rwindow]
        W_ltxy = self.Y_ltxy[Rwindow]
        W_lvel = self.Y_lvel[Rwindow]
        W_lvrt = self.Y_lvrt[Rwindow]
        W_gaze_pos = self.Y_gaze_pos[Rwindow]

        if self.style_encoding_type == "label":
            style = Rlabel
        elif self.style_encoding_type == "example":
            style = self.get_example(Rwindow, sample_range, self.example_window_length)

        return (
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
            style,
        )

    def get_shapes(self):
        num_audio_features = self.X_audio_features.shape[1]
        pose_input_size = len(self.anim_input_std)
        pose_output_size = len(self.anim_output_std)
        dimensions = dict(
            num_audio_features=num_audio_features,
            pose_input_size=pose_input_size,
            pose_output_size=pose_output_size,
        )
        return dimensions

    def get_means_stds(self, device):
        return (
            self.audio_input_mean.to(device),
            self.audio_input_std.to(device),
            self.anim_input_mean.to(device),
            self.anim_input_std.to(device),
            self.anim_output_mean.to(device),
            self.anim_output_std.to(device),
        )

    def get_example(
            self, Rwindow, sample_range, example_window_length,
    ):

        ext_window = (example_window_length - self.window) // 2
        ws = min(ext_window, Rwindow[0] - sample_range[0])
        we = min(ext_window, sample_range[1] - Rwindow[-1])
        s_ext = ws + ext_window - we
        w_ext = we + ext_window - ws
        start = max(Rwindow[0] - s_ext, sample_range[0])
        end = min(Rwindow[-1] + w_ext, sample_range[1]) + 1
        end = min(end, len(self.Y_root_vel))
        S_root_vel = self.Y_root_vel[start:end].reshape(end - start, -1)
        S_root_vrt = self.Y_root_vrt[start:end].reshape(end - start, -1)
        S_lpos = self.Y_lpos[start:end].reshape(end - start, -1)
        S_ltxy = self.Y_ltxy[start:end].reshape(end - start, -1)
        S_lvel = self.Y_lvel[start:end].reshape(end - start, -1)
        S_lvrt = self.Y_lvrt[start:end].reshape(end - start, -1)
        example_feature_vec = torch.cat(
            [S_root_vel, S_root_vrt, S_lpos, S_ltxy, S_lvel, S_lvrt, torch.zeros_like(S_root_vel), ],
            dim=1,
        )
        curr_len = len(example_feature_vec)
        if curr_len < example_window_length:
            example_feature_vec = torch.cat(
                [example_feature_vec, example_feature_vec[-example_window_length + curr_len:]],
                dim=0,
            )
        return example_feature_vec

    def get_sample(self, dataset, length=None, range_index=None):
        if dataset == "train":
            if range_index is None:
                range_index = np.random.randint(len(self.ranges_train))
            (s, e), label = self.ranges_train[range_index], self.ranges_train_labels[range_index]
        elif dataset == "valid":
            if range_index is None:
                range_index = np.random.randint(len(self.ranges_valid))
            (s, e), label = self.ranges_valid[range_index], self.ranges_valid_labels[range_index]

        if length is not None:
            e = min(s + length * 60, e)

        return (
            self.X_audio_features[s:e][np.newaxis],
            self.Y_root_pos[s:e][np.newaxis],
            self.Y_root_rot[s:e][np.newaxis],
            self.Y_root_vel[s:e][np.newaxis],
            self.Y_root_vrt[s:e][np.newaxis],
            self.Y_lpos[s:e][np.newaxis],
            self.Y_ltxy[s:e][np.newaxis],
            self.Y_lvel[s:e][np.newaxis],
            self.Y_lvrt[s:e][np.newaxis],
            self.Y_gaze_pos[s:e][np.newaxis],
            label,
            [s, e],
            range_index,
        )

    def get_stats(self):
        from rich.console import Console
        from rich.table import Table

        console = Console(record=True)
        # Style infos
        df = pd.DataFrame()
        df["Dataset"] = ["Train", "Validation", "Total"]
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        table = Table(title="Data Info", show_lines=True, row_styles=["magenta"])
        table.add_column("Dataset")
        data_len = 0
        for i in range(self.nlabels):
            ind_mask = self.ranges_train_labels == i
            ranges = self.ranges_train[ind_mask]
            num_train_frames = (
                    np.sum(ranges[:, 1] - ranges[:, 0]) / 2
            )  # It is divided by two as we have mirrored versions too
            ind_mask = self.ranges_valid_labels == i
            ranges = self.ranges_valid[ind_mask]
            num_valid_frames = np.sum(ranges[:, 1] - ranges[:, 0]) / 2
            total = num_train_frames + num_valid_frames
            df[self.label_names[i]] = [
                f"{num_train_frames} frames - {num_train_frames / 60:.1f} secs",
                f"{num_valid_frames} frames - {num_valid_frames / 60:.1f} secs",
                f"{total} frames - {total / 60:.1f} secs",
            ]
            table.add_column(self.label_names[i])
            data_len += total

        for i in range(3):
            table.add_row(*list(df.iloc[i]))
        console.print(table)
        dimensions = self.get_shapes()
        console.print(f"Total length of dataset is {data_len} frames - {data_len / 60:.1f} seconds")
        console.print("Num features: ", dimensions)
