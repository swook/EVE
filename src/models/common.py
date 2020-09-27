"""Copyright 2020 ETH Zurich, Seonwook Park

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from core import DefaultConfig

config = DefaultConfig()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pitchyaw_to_vector(a):
    if a.shape[1] == 2:
        sin = torch.sin(a)
        cos = torch.cos(a)
        return torch.stack([cos[:, 0] * sin[:, 1], sin[:, 0], cos[:, 0] * cos[:, 1]], dim=1)
    elif a.shape[1] == 3:
        return F.normalize(a)
    else:
        raise ValueError('Do not know how to convert tensor of size %s' % a.shape)


def vector_to_pitchyaw(a):
    if a.shape[1] == 2:
        return a
    elif a.shape[1] == 3:
        a = a.view(-1, 3)
        norm_a = torch.div(a, torch.norm(a, dim=1).view(-1, 1) + 1e-7)
        return torch.stack([
            torch.asin(norm_a[:, 1]),
            torch.atan2(norm_a[:, 0], norm_a[:, 2]),
        ], dim=1)
    else:
        raise ValueError('Do not know how to convert tensor of size %s' % a.shape)


def pitchyaw_to_rotation(a):
    if a.shape[1] == 3:
        a = vector_to_pitchyaw(a)

    cos = torch.cos(a)
    sin = torch.sin(a)
    ones = torch.ones_like(cos[:, 0])
    zeros = torch.zeros_like(cos[:, 0])
    matrices_1 = torch.stack([ones, zeros, zeros,
                              zeros, cos[:, 0], sin[:, 0],
                              zeros, -sin[:, 0], cos[:, 0]
                              ], dim=1)
    matrices_2 = torch.stack([cos[:, 1], zeros, sin[:, 1],
                              zeros, ones, zeros,
                              -sin[:, 1], zeros, cos[:, 1]
                              ], dim=1)
    matrices_1 = matrices_1.view(-1, 3, 3)
    matrices_2 = matrices_2.view(-1, 3, 3)
    matrices = torch.matmul(matrices_2, matrices_1)
    return matrices


def rotation_to_vector(a):
    assert(a.ndim == 3)
    assert(a.shape[1] == a.shape[2] == 3)
    frontal_vector = torch.cat([
        torch.zeros_like(a[:, :2, 0]).reshape(-1, 2, 1),
        torch.ones_like(a[:, 2, 0]).reshape(-1, 1, 1),
    ], axis=1)
    return torch.matmul(a, frontal_vector)


def apply_transformation(T, vec):
    if vec.shape[1] == 2:
        vec = pitchyaw_to_vector(vec)
    vec = vec.reshape(-1, 3, 1)
    h_vec = F.pad(vec, pad=(0, 0, 0, 1), value=1.0)
    return torch.matmul(T, h_vec)[:, :3, 0]


def apply_rotation(T, vec):
    if vec.shape[1] == 2:
        vec = pitchyaw_to_vector(vec)
    vec = vec.reshape(-1, 3, 1)
    R = T[:, :3, :3]
    return torch.matmul(R, vec).reshape(-1, 3)


nn_plane_normal = None
nn_plane_other = None


def get_intersect_with_zero(o, g):
    """Intersects a given gaze ray (origin o and direction g) with z = 0."""
    global nn_plane_normal, nn_plane_other
    if nn_plane_normal is None:
        nn_plane_normal = torch.tensor([0, 0, 1], dtype=torch.float32, device=device).view(1, 3, 1)
        nn_plane_other = torch.tensor([1, 0, 0], dtype=torch.float32, device=device).view(1, 3, 1)

    # Define plane to intersect with
    n = nn_plane_normal
    a = nn_plane_other
    g = g.view(-1, 3, 1)
    o = o.view(-1, 3, 1)
    numer = torch.sum(torch.mul(a - o, n), dim=1)

    # Intersect with plane using provided 3D origin
    denom = torch.sum(torch.mul(g, n), dim=1) + 1e-7
    t = torch.div(numer, denom).view(-1, 1, 1)
    return (o + torch.mul(t, g))[:, :2, 0]


def calculate_combined_gaze_direction(avg_origin, avg_PoG, head_rotation,
                                      camera_transformation):
    # NOTE: PoG is assumed to be in mm and in the screen-plane
    avg_PoG_3D = F.pad(avg_PoG, (0, 1))

    # Bring to camera-specific coordinate system, where the origin is
    avg_PoG_3D = apply_transformation(camera_transformation, avg_PoG_3D)
    direction = avg_PoG_3D - avg_origin

    # Rotate gaze vector back
    direction = direction.reshape(-1, 3, 1)
    direction = torch.matmul(head_rotation, direction)

    # Negate gaze vector back (to user perspective)
    direction = -direction

    direction = vector_to_pitchyaw(direction)
    return direction


def to_screen_coordinates(origin, direction, rotation, reference_dict):
    direction = pitchyaw_to_vector(direction)

    # Negate gaze vector back (to camera perspective)
    direction = -direction

    # De-rotate gaze vector
    inv_rotation = torch.transpose(rotation, 1, 2)
    direction = direction.reshape(-1, 3, 1)
    direction = torch.matmul(inv_rotation, direction)

    # Transform values
    inv_camera_transformation = reference_dict['inv_camera_transformation']
    direction = apply_rotation(inv_camera_transformation, direction)
    origin = apply_transformation(inv_camera_transformation, origin)

    # Intersect with z = 0
    recovered_target_2D = get_intersect_with_zero(origin, direction)
    PoG_mm = recovered_target_2D

    # Convert back from mm to pixels
    ppm_w = reference_dict['pixels_per_millimeter'][:, 0]
    ppm_h = reference_dict['pixels_per_millimeter'][:, 1]
    PoG_px = torch.stack([
        torch.clamp(recovered_target_2D[:, 0] * ppm_w,
                    0.0, float(config.actual_screen_size[0])),
        torch.clamp(recovered_target_2D[:, 1] * ppm_h,
                    0.0, float(config.actual_screen_size[1]))
    ], axis=-1)

    return PoG_mm, PoG_px


def apply_offset_augmentation(gaze_direction, head_rotation, kappa, inverse_kappa=False):
    gaze_direction = pitchyaw_to_vector(gaze_direction)

    # Negate gaze vector back (to camera perspective)
    gaze_direction = -gaze_direction

    # De-rotate gaze vector
    inv_head_rotation = torch.transpose(head_rotation, 1, 2)
    gaze_direction = gaze_direction.reshape(-1, 3, 1)
    gaze_direction = torch.matmul(inv_head_rotation, gaze_direction)

    # Negate gaze vector back (to user perspective)
    gaze_direction = -gaze_direction

    # Apply kappa to frontal vector [0 0 1]
    kappa_vector = pitchyaw_to_vector(kappa).reshape(-1, 3, 1)
    if inverse_kappa:
        kappa_vector = torch.cat([
            -kappa_vector[:, :2, :], kappa_vector[:, 2, :].reshape(-1, 1, 1),
        ], axis=1)

    # Apply head-relative gaze to rotated frontal vector
    head_relative_gaze_rotation = pitchyaw_to_rotation(vector_to_pitchyaw(gaze_direction))
    gaze_direction = torch.matmul(head_relative_gaze_rotation, kappa_vector)

    # Negate gaze vector back (to camera perspective)
    gaze_direction = -gaze_direction

    # Rotate gaze vector back
    gaze_direction = gaze_direction.reshape(-1, 3, 1)
    gaze_direction = torch.matmul(head_rotation, gaze_direction)

    # Negate gaze vector back (to user perspective)
    gaze_direction = -gaze_direction

    gaze_direction = vector_to_pitchyaw(gaze_direction)
    return gaze_direction


heatmap_xs = None
heatmap_ys = None
heatmap_alpha = None


def make_heatmap(centre, sigma):
    global heatmap_xs, heatmap_ys, heatmap_alpha
    w, h = config.gaze_heatmap_size
    if heatmap_xs is None:
        xs = np.arange(0, w, step=1, dtype=np.float32)
        ys = np.expand_dims(np.arange(0, h, step=1, dtype=np.float32), -1)
        heatmap_xs = torch.tensor(xs).to(device)
        heatmap_ys = torch.tensor(ys).to(device)
    heatmap_alpha = -0.5 / (sigma ** 2)
    cx = (w / config.actual_screen_size[0]) * centre[0]
    cy = (h / config.actual_screen_size[1]) * centre[1]
    heatmap = torch.exp(heatmap_alpha * ((heatmap_xs - cx)**2 + (heatmap_ys - cy)**2))
    heatmap = 1e-8 + heatmap  # Make the zeros non-zero (remove collapsing issue)
    return heatmap.unsqueeze(0)  # make it (1 x H x W) in shape


def batch_make_heatmaps(centres, sigma):
    return torch.stack([make_heatmap(centre, sigma) for centre in centres], axis=0)


gaze_history_map_decay_per_ms = None


def make_gaze_history_map(history_timestamps, heatmaps, validities):
    # NOTE: heatmaps has dimensions T x H x W
    global gaze_history_map_decay_per_ms
    target_timestamp = history_timestamps[torch.nonzero(history_timestamps)][-1]
    output_heatmap = torch.zeros_like(heatmaps[0])
    if gaze_history_map_decay_per_ms is None:
        gaze_history_map_decay_per_ms = \
            torch.tensor(config.gaze_history_map_decay_per_ms).to(device)

    for timestamp, heatmap, validity in zip(history_timestamps, heatmaps, validities):

        if timestamp == 0:
            continue

        # Calculate difference in time in milliseconds
        diff_timestamp = (target_timestamp - timestamp) * 1e-6
        assert(diff_timestamp >= 0)

        # Weights for later weighted average
        time_based_weight = torch.pow(gaze_history_map_decay_per_ms, diff_timestamp).view(1, 1)

        # Keep if within time window
        output_heatmap = output_heatmap + validity.float() * time_based_weight.detach() * heatmap

    return output_heatmap


def batch_make_gaze_history_maps(history_timestamps, heatmaps, validity):
    # NOTE: timestamps is a tensor, heatmaps is a list of tensors
    batch_size = history_timestamps.shape[0]
    history_len = len(heatmaps)
    return torch.stack([
        make_gaze_history_map(
            history_timestamps[b, :history_len],
            [h[b, :] for h in heatmaps],
            validity[b, :history_len],
        )
        for b in range(batch_size)
    ], axis=0)


softargmax_xs = None
softargmax_ys = None


def soft_argmax(heatmaps):
    global softargmax_xs, softargmax_ys
    if softargmax_xs is None:
        # Assume normalized coordinate [0, 1] for numeric stability
        w, h = config.gaze_heatmap_size
        ref_xs, ref_ys = np.meshgrid(np.linspace(0, 1.0, num=w, endpoint=True),
                                     np.linspace(0, 1.0, num=h, endpoint=True),
                                     indexing='xy')
        ref_xs = np.reshape(ref_xs, [1, h*w])
        ref_ys = np.reshape(ref_ys, [1, h*w])
        softargmax_xs = torch.tensor(ref_xs.astype(np.float32)).to(device)
        softargmax_ys = torch.tensor(ref_ys.astype(np.float32)).to(device)
    ref_xs, ref_ys = softargmax_xs, softargmax_ys

    # Yield softmax+integrated coordinates in [0, 1]
    n, _, h, w = heatmaps.shape
    assert(w == config.gaze_heatmap_size[0])
    assert(h == config.gaze_heatmap_size[1])
    beta = 1e2
    x = heatmaps.view(-1, h*w)
    x = F.softmax(beta * x, dim=-1)
    lmrk_xs = torch.sum(ref_xs * x, axis=-1)
    lmrk_ys = torch.sum(ref_ys * x, axis=-1)

    # Return to actual coordinates ranges
    pixel_xs = torch.clamp(config.actual_screen_size[0] * lmrk_xs,
                           0.0, config.actual_screen_size[0])
    pixel_ys = torch.clamp(config.actual_screen_size[1] * lmrk_ys,
                           0.0, config.actual_screen_size[1])
    return torch.stack([pixel_xs, pixel_ys], axis=-1)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class CRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CRNNCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cell = nn.Conv2d(self.input_size + self.hidden_size, self.hidden_size,
                              kernel_size=3, padding=1)

    def forward(self, x, previous_states=None):
        batch_size = x.shape[0]
        if previous_states is None:
            state_shape = [batch_size, self.hidden_size] + list(x.shape[2:])
            hidden_state = torch.autograd.Variable(torch.zeros(state_shape)).to(device)
        else:
            hidden_state = previous_states

        # Apply RNN
        hidden = self.cell(torch.cat([x, hidden_state], axis=1))
        hidden = torch.tanh(hidden)
        return hidden


class CLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.gates = nn.Conv2d(self.input_size + self.hidden_size, 4 * self.hidden_size,
                               kernel_size=3, padding=1)

    def forward(self, x, previous_states=None):
        batch_size = x.shape[0]
        if previous_states is None:
            state_shape = [batch_size, self.hidden_size] + list(x.shape[2:])
            hidden_state = torch.autograd.Variable(torch.zeros(state_shape)).to(device)
            cell_state = torch.autograd.Variable(torch.zeros(state_shape)).to(device)
        else:
            hidden_state, cell_state = previous_states

        # Apply LSTM
        gates = self.gates(torch.cat([x, hidden_state], axis=1))
        in_gate, forget_gate, out_gate, cell_gate = gates.chunk(4, 1)
        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        out_gate = torch.sigmoid(out_gate)
        cell_gate = torch.tanh(cell_gate)
        forget = (forget_gate * cell_state)
        update = (in_gate * cell_gate)
        cell = forget + update
        hidden = out_gate * torch.tanh(cell)
        return hidden, cell


class CGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CGRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.gates_1 = nn.Conv2d(self.input_size + self.hidden_size, 2 * self.hidden_size,
                                 kernel_size=3, padding=1)
        self.gate_2 = nn.Conv2d(self.input_size + self.hidden_size, self.hidden_size,
                                kernel_size=3, padding=1)

    def forward(self, x, previous_states=None):
        batch_size = x.shape[0]
        if previous_states is None:
            state_shape = [batch_size, self.hidden_size] + list(x.shape[2:])
            hidden_state = torch.autograd.Variable(torch.zeros(state_shape)).to(device)
        else:
            hidden_state = previous_states

        # Apply GRU
        gates_1 = self.gates_1(torch.cat([x, hidden_state], axis=1))
        reset_gate, update_gate = torch.sigmoid(gates_1).chunk(2, 1)
        reset_gate = (reset_gate * hidden_state)
        output_gate = self.gate_2(torch.cat([reset_gate, x], axis=1))
        output_gate = torch.tanh(output_gate)
        hidden = (1. - update_gate) * output_gate + update_gate * hidden_state
        return hidden
