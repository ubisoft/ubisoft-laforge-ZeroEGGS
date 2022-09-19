import numpy as np
import torch.nn as nn
from torch.nn import functional as F

from anim.tquat import *


# ===============================================
#                   Decoder
# ===============================================
class Decoder(nn.Module):
    def __init__(
            self,
            pose_input_size,
            pose_output_size,
            speech_encoding_size,
            style_encoding_size,
            hidden_size,
            num_rnn_layers,
            rnn_cond="normal",
    ):
        super(Decoder, self).__init__()

        if rnn_cond == "normal":
            self.recurrent_decoder = RecurrentDecoderNormal(
                pose_input_size,
                speech_encoding_size,
                style_encoding_size,
                pose_output_size,
                hidden_size,
                num_rnn_layers,
            )
        elif rnn_cond == "film":
            self.recurrent_decoder = RecurrentDecoderFiLM(
                pose_input_size,
                speech_encoding_size,
                style_encoding_size,
                pose_output_size,
                hidden_size,
                num_rnn_layers,
            )

        self.cell_state_encoder = CellStateEncoder(
            pose_input_size + style_encoding_size, hidden_size, num_rnn_layers
        )

    def forward(
            self,
            Z_root_pos,
            Z_root_rot,
            Z_root_vel,
            Z_root_vrt,
            Z_lpos,
            Z_ltxy,
            Z_lvel,
            Z_lvrt,
            Z_gaze_pos,
            speech_encoding,
            style_encoding,
            parents,
            anim_input_mean,
            anim_input_std,
            anim_output_mean,
            anim_output_std,
            dt: float,
    ):

        batchsize = speech_encoding.shape[0]
        nframes = speech_encoding.shape[1]

        # Getting initial values from ground truth
        O_root_pos = [Z_root_pos]
        O_root_rot = [Z_root_rot]
        O_root_vel = [Z_root_vel]
        O_root_vrt = [Z_root_vrt]
        O_lpos = [Z_lpos]
        O_ltxy = [Z_ltxy]
        O_lvel = [Z_lvel]
        O_lvrt = [Z_lvrt]

        # Initialize the hidden state of decoder
        decoder_state = self.cell_state_encoder(
            vectorize_input(
                Z_root_pos,
                Z_root_rot,
                Z_root_vel,
                Z_root_vrt,
                Z_lpos,
                Z_ltxy,
                Z_lvel,
                Z_lvrt,
                Z_gaze_pos[:, 0],
                parents,
                anim_input_mean,
                anim_input_std,
            ),
            style_encoding[:, 0],
        )

        for i in range(1, nframes):
            # Prepare Input
            pose_encoding = vectorize_input(
                O_root_pos[-1],
                O_root_rot[-1],
                O_root_vel[-1],
                O_root_vrt[-1],
                O_lpos[-1],
                O_ltxy[-1],
                O_lvel[-1],
                O_lvrt[-1],
                Z_gaze_pos[:, i],
                parents,
                anim_input_mean,
                anim_input_std,
            )

            # Predict
            predicted, decoder_state = self.recurrent_decoder(
                pose_encoding, speech_encoding[:, i], style_encoding[:, i], decoder_state
            )

            # Integrate Prediction
            (
                P_root_pos,
                P_root_rot,
                P_root_vel,
                P_root_vrt,
                P_lpos,
                P_ltxy,
                P_lvel,
                P_lvrt,
            ) = devectorize_output(
                predicted,
                O_root_pos[-1],
                O_root_rot[-1],
                Z_lpos.shape[0],
                Z_lpos.shape[1],
                dt,
                anim_output_mean,
                anim_output_std,
            )

            # Append
            O_root_pos.append(P_root_pos)
            O_root_rot.append(P_root_rot)
            O_root_vel.append(P_root_vel)
            O_root_vrt.append(P_root_vrt)
            O_lpos.append(P_lpos)
            O_ltxy.append(P_ltxy)
            O_lvel.append(P_lvel)
            O_lvrt.append(P_lvrt)

        return (
            torch.cat([O[:, None] for O in O_root_pos], dim=1),
            torch.cat([O[:, None] for O in O_root_rot], dim=1),
            torch.cat([O[:, None] for O in O_root_vel], dim=1),
            torch.cat([O[:, None] for O in O_root_vrt], dim=1),
            torch.cat([O[:, None] for O in O_lpos], dim=1),
            torch.cat([O[:, None] for O in O_ltxy], dim=1),
            torch.cat([O[:, None] for O in O_lvel], dim=1),
            torch.cat([O[:, None] for O in O_lvrt], dim=1),
        )


class RecurrentDecoderNormal(nn.Module):
    def __init__(
            self, pose_input_size, speech_size, style_size, output_size, hidden_size, num_rnn_layers
    ):
        super(RecurrentDecoderNormal, self).__init__()

        all_input_size = pose_input_size + speech_size + style_size
        self.layer0 = nn.Linear(all_input_size, hidden_size)
        self.layer1 = nn.GRU(
            all_input_size + hidden_size, hidden_size, num_rnn_layers, batch_first=True
        )

        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, pose, speech, style, cell_state):
        hidden = F.elu(self.layer0(torch.cat([pose, speech, style], dim=-1)))
        cell_output, cell_state = self.layer1(
            torch.cat([hidden, pose, speech, style], dim=-1).unsqueeze(1), cell_state
        )
        output = self.layer2(cell_output.squeeze(1))
        return output, cell_state


class RecurrentDecoderFiLM(nn.Module):
    def __init__(
            self, pose_input_size, speech_size, style_size, output_size, hidden_size, num_rnn_layers
    ):
        super(RecurrentDecoderFiLM, self).__init__()

        self.hidden_size = hidden_size
        self.gammas_predictor = LinearNorm(
            style_size, hidden_size * 2, w_init_gain="linear"
        )
        self.betas_predictor = LinearNorm(
            style_size, hidden_size * 2, w_init_gain="linear"
        )

        all_input_size = pose_input_size + speech_size + style_size
        self.layer0 = nn.Linear(pose_input_size + speech_size, hidden_size)
        self.layer1 = nn.GRU(
            pose_input_size + speech_size + hidden_size,
            hidden_size,
            num_rnn_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, pose, speech, style, cell_state):
        gammas = self.gammas_predictor(style)
        gammas = gammas + 1
        betas = self.betas_predictor(style)

        hidden = F.elu(self.layer0(torch.cat([pose, speech], dim=-1)))
        hidden = hidden * gammas[:, : self.hidden_size] + betas[:, : self.hidden_size]
        cell_output, cell_state = self.layer1(
            torch.cat([hidden, pose, speech], dim=-1).unsqueeze(1), cell_state
        )
        hidden = F.elu(self.layer2(cell_output.squeeze(1)))
        hidden = hidden * gammas[:, self.hidden_size:] + betas[:, self.hidden_size:]
        output = self.layer3(hidden)
        return output, cell_state


class CellStateEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_rnn_layers):
        super(CellStateEncoder, self).__init__()
        self.num_rnn_layers = num_rnn_layers
        self.layer0 = nn.Linear(input_size, hidden_size)
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size * num_rnn_layers)

    def forward(self, pose, style):
        hidden = F.elu(self.layer0(torch.cat([pose, style], dim=-1)))
        hidden = F.elu(self.layer1(hidden))
        output = self.layer2(hidden)

        return output.reshape(output.shape[0], self.num_rnn_layers, -1).swapaxes(0, 1).contiguous()


# ===============================================
#                   Speech Encoder
# ===============================================
class SpeechEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SpeechEncoder, self).__init__()

        self.layer0 = nn.Conv1d(
            input_size, hidden_size, kernel_size=1, padding="same", padding_mode="replicate"
        )
        self.drop0 = nn.Dropout(p=0.2)

        self.layer1 = nn.Conv1d(
            hidden_size, output_size, kernel_size=31, padding="same", padding_mode="replicate"
        )
        self.drop1 = nn.Dropout(p=0.2)

        self.layer2 = nn.Linear(output_size, output_size)

    def forward(self, x):
        x = torch.swapaxes(x, 1, 2)
        x = self.drop0(F.elu(self.layer0(x)))
        x = self.drop1(F.elu(self.layer1(x)))
        x = torch.swapaxes(x, 1, 2)
        x = F.elu(self.layer2(x))

        return x


# ===============================================
#                   Style Encoder
# ===============================================
class StyleEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, style_embedding_size, type="attn", use_vae=False):
        super(StyleEncoder, self).__init__()
        self.use_vae = use_vae
        self.style_embedding_size = style_embedding_size
        output_size = 2 * style_embedding_size if use_vae else style_embedding_size
        if type == "gru":
            self.encoder = StyleEncoderGRU(input_size, hidden_size, output_size)
        elif type == "attn":
            self.encoder = StyleEncoderAttn(input_size, hidden_size, output_size)

    def forward(self, input, temprature: float = 1.0):
        encoder_output = self.encoder(input)
        if self.use_vae:
            mu, logvar = (
                encoder_output[:, : self.style_embedding_size],
                encoder_output[:, self.style_embedding_size:],
            )

            # re-parameterization trick
            std = torch.exp(0.5 * logvar) / temprature
            eps = torch.randn_like(std)

            style_embedding = mu + eps * std
            return style_embedding, mu, logvar
        else:
            return encoder_output, None, None


class StyleEncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, style_embedding_size):
        super(StyleEncoderGRU, self).__init__()

        self.convs = nn.Sequential(
            ConvNorm1D(
                input_size,
                hidden_size,
                kernel_size=3,
                stride=1,
                padding=int((3 - 1) / 2),
                dilation=1,
                w_init_gain="relu",
            ),
            nn.ReLU(),
            # AvgPoolNorm1D(kernel_size=2),
            ConvNorm1D(
                hidden_size,
                hidden_size,
                kernel_size=3,
                stride=1,
                padding=int((3 - 1) / 2),
                dilation=1,
                w_init_gain="relu",
            ),
            nn.ReLU(),
        )
        self.rnn_layer = nn.GRU(hidden_size, hidden_size, 1, batch_first=True, bidirectional=True)
        self.projection_layer = LinearNorm(
            hidden_size * 2, style_embedding_size, w_init_gain="linear"
        )

    def forward(self, input):
        input = self.convs(input)
        output, _ = self.rnn_layer(input)
        style_embedding = self.projection_layer(output[:, -1])
        return style_embedding


class StyleEncoderAttn(nn.Module):
    """ Style Encoder Module:
        - Positional Encoding
        - Nf x FFT Blocks
        - Linear Projection Layer
    """

    def __init__(self, input_size, hidden_size, style_embedding_size):
        super(StyleEncoderAttn, self).__init__()

        # positional encoding
        self.pos_enc = PositionalEncoding(style_embedding_size)

        self.convs = nn.Sequential(
            ConvNorm1D(
                input_size,
                hidden_size,
                kernel_size=3,
                stride=1,
                padding=int((3 - 1) / 2),
                dilation=1,
                w_init_gain="relu",
            ),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.2),
            ConvNorm1D(
                hidden_size,
                style_embedding_size,
                kernel_size=3,
                stride=1,
                padding=int((3 - 1) / 2),
                dilation=1,
                w_init_gain="relu",
            ),
            nn.ReLU(),
            nn.LayerNorm(style_embedding_size),
            nn.Dropout(0.2),
        )
        # FFT blocks
        blocks = []
        for _ in range(1):
            blocks.append(FFTBlock(style_embedding_size))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, input):
        """ Forward function of Prosody Encoder:
            frames_energy = (B, T_max)
            frames_pitch = (B, T_max)
            mel_specs = (B, nb_mels, T_max)
            speaker_ids = (B, )
            output_lengths = (B, )
        """
        output_lengths = torch.as_tensor(
            len(input) * [input.shape[1]], device=input.device, dtype=torch.int32
        )
        # compute positional encoding
        pos = self.pos_enc(output_lengths.unsqueeze(1)).to(input.device)  # (B, T_max, hidden_embed_dim)
        # pass through convs
        outputs = self.convs(input)  # (B, T_max, hidden_embed_dim)

        # create mask
        mask = ~get_mask_from_lengths(output_lengths)  # (B, T_max)
        # add encodings and mask tensor
        outputs = outputs + pos  # (B, T_max, hidden_embed_dim)
        outputs = outputs.masked_fill(mask.unsqueeze(2), 0)  # (B, T_max, hidden_embed_dim)
        # pass through FFT blocks
        for _, block in enumerate(self.blocks):
            outputs = block(outputs, None, mask)  # (B, T_max, hidden_embed_dim)
        # average pooling on the whole time sequence
        style_embedding = torch.sum(outputs, dim=1) / output_lengths.unsqueeze(
            1
        )  # (B, hidden_embed_dim)

        return style_embedding


# ===============================================
#                   Sub-modules
# ===============================================
class LinearNorm(nn.Module):
    """ Linear Norm Module:
        - Linear Layer
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        """ Forward function of Linear Norm
            x = (*, in_dim)
        """
        x = self.linear_layer(x)  # (*, out_dim)

        return x


class PositionalEncoding(nn.Module):
    """ Positional Encoding Module:
        - Sinusoidal Positional Embedding
    """

    def __init__(self, embed_dim, max_len=20000, timestep=10000.0):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2).float() * (-np.log(timestep) / self.embed_dim)
        )  # (embed_dim // 2, )
        self.pos_enc = torch.FloatTensor(max_len, self.embed_dim).zero_()  # (max_len, embed_dim)
        self.pos_enc[:, 0::2] = torch.sin(pos * div_term)
        self.pos_enc[:, 1::2] = torch.cos(pos * div_term)

    def forward(self, x):
        """ Forward function of Positional Encoding:
            x = (B, N) -- Long or Int tensor
        """
        # initialize tensor
        nb_frames_max = torch.max(torch.cumsum(x, dim=1))
        pos_emb = torch.FloatTensor(
            x.size(0), nb_frames_max, self.embed_dim
        ).zero_()  # (B, nb_frames_max, embed_dim)
        # pos_emb = pos_emb.cuda(x.device, non_blocking=True).float()  # (B, nb_frames_max, embed_dim)

        # TODO: Check if we can remove the for loops
        for line_idx in range(x.size(0)):
            pos_idx = []
            for column_idx in range(x.size(1)):
                idx = x[line_idx, column_idx]
                pos_idx.extend([i for i in range(idx)])
            emb = self.pos_enc[pos_idx]  # (nb_frames, embed_dim)
            pos_emb[line_idx, : emb.size(0), :] = emb

        return pos_emb


class FFTBlock(nn.Module):
    """ FFT Block Module:
        - Multi-Head Attention
        - Position Wise Convolutional Feed-Forward
        - FiLM conditioning (if film_params is not None)
    """

    def __init__(self, hidden_size):
        super(FFTBlock, self).__init__()
        self.attention = MultiHeadAttention(hidden_size)
        self.feed_forward = PositionWiseConvFF(hidden_size)

    def forward(self, x, film_params, mask):
        """ Forward function of FFT Block:
            x = (B, L_max, hidden_embed_dim)
            film_params = (B, nb_film_params)
            mask = (B, L_max)
        """
        # attend
        attn_outputs, _ = self.attention(
            x, x, x, key_padding_mask=mask
        )  # (B, L_max, hidden_embed_dim)
        attn_outputs = attn_outputs.masked_fill(
            mask.unsqueeze(2), 0
        )  # (B, L_max, hidden_embed_dim)
        # feed-forward pass
        outputs = self.feed_forward(attn_outputs, film_params)  # (B, L_max, hidden_embed_dim)
        outputs = outputs.masked_fill(mask.unsqueeze(2), 0)  # (B, L_max, hidden_embed_dim)

        return outputs


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention Module:
        - Multi-Head Attention
            A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser and I. Polosukhin
            "Attention is all you need",
            in NeurIPS, 2017.
        - Dropout
        - Residual Connection 
        - Layer Normalization
    """

    def __init__(self, hidden_size):
        super(MultiHeadAttention, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(hidden_size, 4, 0.1)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """ Forward function of Multi-Head Attention:
            query = (B, L_max, hidden_embed_dim)
            key = (B, T_max, hidden_embed_dim)
            value = (B, T_max, hidden_embed_dim)
            key_padding_mask = (B, T_max) if not None
            attn_mask = (L_max, T_max) if not None
        """
        # compute multi-head attention
        # attn_outputs = (L_max, B, hidden_embed_dim)
        # attn_weights = (B, L_max, T_max)
        attn_outputs, attn_weights = self.multi_head_attention(
            query.transpose(0, 1),
            key.transpose(0, 1),
            value.transpose(0, 1),
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        attn_outputs = attn_outputs.transpose(0, 1)  # (B, L_max, hidden_embed_dim)
        # apply dropout
        attn_outputs = self.dropout(attn_outputs)  # (B, L_max, hidden_embed_dim)
        # add residual connection and perform layer normalization
        attn_outputs = self.layer_norm(attn_outputs + query)  # (B, L_max, hidden_embed_dim)

        return attn_outputs, attn_weights


class PositionWiseConvFF(nn.Module):
    """ Position Wise Convolutional Feed-Forward Module:
        - 2x Conv 1D with ReLU
        - Dropout
        - Residual Connection 
        - Layer Normalization
        - FiLM conditioning (if film_params is not None)
    """

    def __init__(self, hidden_size):
        super(PositionWiseConvFF, self).__init__()
        self.convs = nn.Sequential(
            ConvNorm1D(
                hidden_size,
                hidden_size,
                kernel_size=3,
                stride=1,
                padding=int((3 - 1) / 2),
                dilation=1,
                w_init_gain="relu",
            ),
            nn.ReLU(),
            ConvNorm1D(
                hidden_size,
                hidden_size,
                kernel_size=3,
                stride=1,
                padding=int((3 - 1) / 2),
                dilation=1,
                w_init_gain="linear",
            ),
            nn.Dropout(0.1),
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, film_params):
        """ Forward function of PositionWiseConvFF:
            x = (B, L_max, hidden_embed_dim)
            film_params = (B, nb_film_params)
        """
        # pass through convs
        outputs = self.convs(x)  # (B, L_max, hidden_embed_dim)
        # add residual connection and perform layer normalization
        outputs = self.layer_norm(outputs + x)  # (B, L_max, hidden_embed_dim)
        # add FiLM transformation
        if film_params is not None:
            nb_gammas = int(film_params.size(1) / 2)
            assert nb_gammas == outputs.size(2)
            gammas = film_params[:, :nb_gammas].unsqueeze(1)  # (B, 1, hidden_embed_dim)
            betas = film_params[:, nb_gammas:].unsqueeze(1)  # (B, 1, hidden_embed_dim)
            outputs = gammas * outputs + betas  # (B, L_max, hidden_embed_dim)

        return outputs


class ConvNorm1D(nn.Module):
    """ Conv Norm 1D Module:
        - Conv 1D
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=None,
            dilation=1,
            bias=True,
            w_init_gain="linear",
    ):
        super(ConvNorm1D, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        """ Forward function of Conv Norm 1D
            x = (B, L, in_channels)
        """
        x = x.transpose(1, 2)  # (B, in_channels, L)
        x = self.conv(x)  # (B, out_channels, L)
        x = x.transpose(1, 2)  # (B, L, out_channels)

        return x


class AvgPoolNorm1D(nn.Module):
    def __init__(
            self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True
    ):
        super(AvgPoolNorm1D, self).__init__()
        self.avgpool1d = nn.AvgPool1d(kernel_size, stride, padding, ceil_mode, count_include_pad)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, in_channels, L)
        x = self.avgpool1d(x)  # (B, out_channels, L)
        x = x.transpose(1, 2)  # (B, L, out_channels)

        return x


# ===============================================
#                   Funcs
# ===============================================
@torch.jit.script
def normalize(x, eps: float = 1e-8):
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


@torch.jit.script
def vectorize_input(
        Z_root_pos,
        Z_root_rot,
        Z_root_vel,
        Z_root_vrt,
        Z_lpos,
        Z_ltxy,
        Z_lvel,
        Z_lvrt,
        Z_gaze_pos,
        parents,
        anim_input_mean,
        anim_input_std,
):
    batchsize = Z_lpos.shape[0]

    # Compute Local Gaze
    # Z_gaze_dir = quat_inv_mul_vec(Z_root_rot, normalize(Z_gaze_pos - Z_root_pos))
    Z_gaze_dir = quat_inv_mul_vec(Z_root_rot, Z_gaze_pos - Z_root_pos)

    # Flatten the autoregressive input
    pose_encoding = torch.cat(
        [
            Z_root_vel.reshape([batchsize, -1]),
            Z_root_vrt.reshape([batchsize, -1]),
            Z_lpos.reshape([batchsize, -1]),
            Z_ltxy.reshape([batchsize, -1]),
            Z_lvel.reshape([batchsize, -1]),
            Z_lvrt.reshape([batchsize, -1]),
            Z_gaze_dir.reshape([batchsize, -1]),
        ],
        dim=1,
    )

    # Normalize
    return (pose_encoding - anim_input_mean) / anim_input_std


@torch.jit.script
def devectorize_output(
        predicted,
        Z_root_pos,
        Z_root_rot,
        batchsize: int,
        njoints: int,
        dt: float,
        anim_output_mean,
        anim_output_std,
):
    # Denormalize
    predicted = (predicted * anim_output_std) + anim_output_mean

    # Extract predictions
    P_root_vel = predicted[:, 0:3]
    P_root_vrt = predicted[:, 3:6]
    P_lpos = predicted[:, 6 + njoints * 0: 6 + njoints * 3].reshape([batchsize, njoints, 3])
    P_ltxy = predicted[:, 6 + njoints * 3: 6 + njoints * 9].reshape([batchsize, njoints, 2, 3])
    P_lvel = predicted[:, 6 + njoints * 9: 6 + njoints * 12].reshape([batchsize, njoints, 3])
    P_lvrt = predicted[:, 6 + njoints * 12: 6 + njoints * 15].reshape([batchsize, njoints, 3])

    # Update pose state
    P_root_pos = quat_mul_vec(Z_root_rot, P_root_vel * dt) + Z_root_pos
    P_root_rot = quat_mul(quat_from_helical(quat_mul_vec(Z_root_rot, P_root_vrt * dt)), Z_root_rot)

    return (P_root_pos, P_root_rot, P_root_vel, P_root_vrt, P_lpos, P_ltxy, P_lvel, P_lvrt)


def generalized_logistic_function(x, center=0.0, B=1.0, A=0.0, K=1.0, C=1.0, Q=1.0, nu=1.0):
    """ Equation of the generalised logistic function
        https://en.wikipedia.org/wiki/Generalised_logistic_function

    :param x:           abscissa point where logistic function needs to be evaluated
    :param center:      abscissa point corresponding to starting time
    :param B:           growth rate
    :param A:           lower asymptote
    :param K:           upper asymptote when C=1.
    :param C:           change upper asymptote value
    :param Q:           related to value at starting time abscissa point
    :param nu:          affects near which asymptote maximum growth occurs

    :return: value of logistic function at abscissa point
    """
    value = A + (K - A) / (C + Q * np.exp(-B * (x - center))) ** (1 / nu)
    return value


def compute_KL_div(mu, logvar, iteration):
    """ Compute KL divergence loss
        mu = (B, embed_dim)
        logvar = (B, embed_dim)
    """
    # compute KL divergence
    # see Appendix B from VAE paper:
    # D.P. Kingma and M. Welling, "Auto-Encoding Variational Bayes", ICLR, 2014.

    kl_weight_center = 7500  # iteration at which weight of KL divergence loss is 0.5
    kl_weight_growth_rate = 0.005  # growth rate for weight of KL divergence loss
    kl_threshold = 2e-1  # KL weight threshold
    # kl_threshold = 1.0

    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # (B, )
    kl_div = torch.mean(kl_div)

    # compute weight for KL cost annealing:
    # S.R. Bowman, L. Vilnis, O. Vinyals, A.M. Dai, R. Jozefowicz, S. Bengio,
    # "Generating Sentences from a Continuous Space", arXiv:1511.06349, 2016.
    kl_div_weight = generalized_logistic_function(
        iteration, center=kl_weight_center, B=kl_weight_growth_rate,
    )
    # apply weight threshold
    kl_div_weight = min(kl_div_weight, kl_threshold)
    return kl_div, kl_div_weight


def compute_kl_uni_gaus(q_params: Tuple, p_params: Tuple):
    mu_q, log_var_q = q_params
    mu_p, log_var_p = p_params

    kl = 0.5 * (log_var_p - log_var_q) + (log_var_q.exp() + (mu_q - mu_p) ** 2) / (2 * log_var_p.exp()) - 0.5 + 1e-8
    kl = torch.sum(kl, dim=-1)
    kl = torch.mean(kl)
    return kl


def get_mask_from_lengths(lengths):
    """ Create a masked tensor from given lengths

    :param lengths:     torch.tensor of size (B, ) -- lengths of each example

    :return mask: torch.tensor of size (B, max_length) -- the masked tensor
    """
    max_len = torch.max(lengths)
    # ids = torch.arange(0, max_len).cuda(lengths.device, non_blocking=True).long()
    ids = torch.arange(0, max_len).long().to(lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool().to(lengths.device)
    return mask
