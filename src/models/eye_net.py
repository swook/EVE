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
import logging

import numpy as np
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, ResNet

from core import DefaultConfig

config = DefaultConfig()
logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
half_pi = 0.5 * np.pi


class EyeNet(nn.Module):
    def __init__(self):
        super(EyeNet, self).__init__()

        num_features = (
            config.eye_net_rnn_num_features
            if config.eye_net_use_rnn
            else config.eye_net_static_num_features
        )

        # CNN backbone (ResNet-18 with instance normalization)
        self.cnn_layers = ResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                                 num_classes=num_features,
                                 norm_layer=nn.InstanceNorm2d)
        self.fc_common = nn.Sequential(
            nn.Linear(num_features + (2 if config.eye_net_use_head_pose_input else 0),
                      num_features),
            nn.SELU(inplace=True),
            nn.Linear(num_features, num_features),
        )

        if config.eye_net_use_rnn:
            # Define RNN cell
            rnn_cells = []
            for i in range(config.eye_net_rnn_num_cells):
                if config.eye_net_rnn_type == 'RNN':
                    rnn_cells.append(nn.RNNCell(input_size=config.eye_net_rnn_num_features,
                                                hidden_size=config.eye_net_rnn_num_features))
                elif config.eye_net_rnn_type == 'LSTM':
                    rnn_cells.append(nn.LSTMCell(input_size=config.eye_net_rnn_num_features,
                                                 hidden_size=config.eye_net_rnn_num_features))
                elif config.eye_net_rnn_type == 'GRU':
                    rnn_cells.append(nn.GRUCell(input_size=config.eye_net_rnn_num_features,
                                                hidden_size=config.eye_net_rnn_num_features))
                else:
                    raise ValueError('Unknown RNN type for EyeNet: %s' % config.eye_net_rnn_type)
            self.rnn_cells = nn.ModuleList(rnn_cells)
        else:
            self.static_fc = nn.Sequential(
                nn.Linear(num_features, num_features),
                nn.SELU(inplace=True),
            )

        # FC layers
        self.fc_to_gaze = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.SELU(inplace=True),
            nn.Linear(num_features, 2, bias=False),
            nn.Tanh(),
        )
        self.fc_to_pupil = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.SELU(inplace=True),
            nn.Linear(num_features, 1),
            nn.ReLU(inplace=True),
        )

        # Set gaze layer weights to zero as otherwise this can
        # explode early in training
        nn.init.zeros_(self.fc_to_gaze[-2].weight)

    def forward(self, input_dict, output_dict, side, previous_output_dict=None):
        # Pick input image
        if (side + '_eye_patch') in output_dict:
            input_image = output_dict[side + '_eye_patch']
        else:
            input_image = input_dict[side + '_eye_patch']

        # Compute CNN features
        initial_features = self.cnn_layers(input_image)

        # Process head pose input if asked for
        if config.eye_net_use_head_pose_input:
            initial_features = torch.cat([initial_features, input_dict[side + '_h']], axis=1)
        initial_features = self.fc_common(initial_features)

        # Apply RNN cells
        if config.eye_net_use_rnn:
            rnn_features = initial_features
            for i, rnn_cell in enumerate(self.rnn_cells):
                suffix = '_%d' % i

                # Retrieve previous hidden/cell states if any
                previous_states = None
                if previous_output_dict is not None:
                    previous_states = previous_output_dict[side + '_eye_rnn_states' + suffix]

                # Inference through RNN cell
                states = rnn_cell(rnn_features, previous_states)

                # Decide what the output is and store back current states
                if isinstance(states, tuple):
                    rnn_features = states[0]
                    output_dict[side + '_eye_rnn_states' + suffix] = states
                else:
                    rnn_features = states
                    output_dict[side + '_eye_rnn_states' + suffix] = states
            features = rnn_features
        else:
            features = self.static_fc(initial_features)

        # Final prediction
        gaze_prediction = half_pi * self.fc_to_gaze(features)
        pupil_size = self.fc_to_pupil(features)

        # For gaze, the range of output values are limited by a tanh and scaling
        output_dict[side + '_g_initial'] = gaze_prediction

        # Estimate of pupil size
        output_dict[side + '_pupil_size'] = pupil_size.reshape(-1)

        # If network frozen, we're gonna detach gradients here
        if config.eye_net_frozen:
            output_dict[side + '_g_initial'] = output_dict[side + '_g_initial'].detach()
