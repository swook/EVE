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
from torch import nn
from torch.nn import functional as F

from core import DefaultConfig
from .common import CRNNCell, CLSTMCell, CGRUCell

config = DefaultConfig()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

half_pi = 0.5 * np.pi


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_shape, out_shape, act_func=nn.ReLU):
        super(BasicBlock, self).__init__()
        ic, ih, iw = in_shape
        oc, oh, ow = out_shape
        assert(ih == oh and iw == ow)

        # Main layers
        self.layers = nn.Sequential(
            nn.InstanceNorm2d(ic, affine=True),
            act_func(inplace=True),
            nn.Conv2d(ic, oc, kernel_size=3, stride=1, padding=1),

            nn.InstanceNorm2d(oc, affine=True),
            act_func(inplace=True),
            nn.Conv2d(oc, oc, kernel_size=3, stride=1, padding=1),
        )

        # Skip layer
        self.skip_layer = None
        if ic != oc:
            self.skip_layer = nn.Sequential(
                nn.InstanceNorm2d(ic, affine=True),
                act_func(inplace=True),
                nn.Conv2d(ic, oc, kernel_size=1, stride=1),
            )

    def forward(self, x, output_dict, previous_output_dict):
        skip = self.skip_layer(x) if self.skip_layer else x
        x = self.layers(x)
        return x + skip


class WrapEncoderDecoder(nn.Module):
    def __init__(self, in_shape, out_shape, module_to_wrap, add_skip_connection=False,
                 num_encoder_blocks=1, num_decoder_blocks=1):
        super(WrapEncoderDecoder, self).__init__()
        ic, ih, iw = in_shape
        oc, oh, ow = out_shape
        assert(ih == oh and iw == ow)
        self.in_shape = in_shape
        self.out_shape = out_shape
        b_ic, bh, bw = module_to_wrap.in_shape
        b_oc = module_to_wrap.out_shape[0]

        self.add_skip_connection = add_skip_connection

        # Define encoder layer blocks
        self.encoder_blocks = nn.ModuleList([BasicBlock([ic, ih, iw], [b_ic, ih, iw])])
        if num_encoder_blocks > 1:
            for _ in range(num_encoder_blocks - 1):
                self.encoder_blocks.append(BasicBlock([b_ic, ih, iw], [b_ic, ih, iw]))

        # Maybe downsample
        self.downsample = None
        if ih != bh or iw != bw:
            self.downsample = nn.AdaptiveMaxPool2d([bh, bw])

        # Reference to in-between module
        self.between_module = module_to_wrap

        # Maybe upsample
        self.upsample = None
        if bh != oh or bw != ow:
            self.upsample = nn.Upsample(size=[oh, ow], mode='bilinear', align_corners=False)

        # Decide on features to concatenate, then decode
        features_to_decode = b_oc
        if add_skip_connection:
            features_to_decode += b_ic
        self.decoder_blocks = nn.ModuleList([
            BasicBlock([features_to_decode, oh, ow], [oc, oh, ow], nn.LeakyReLU)
        ])
        if num_decoder_blocks > 1:
            for _ in range(num_decoder_blocks - 1):
                self.decoder_blocks.append(
                    BasicBlock([oc, oh, ow], [oc, oh, ow], nn.LeakyReLU))

    def forward(self, input_features, output_dict, previous_output_dict):
        x = input_features
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, output_dict, previous_output_dict)
        encoded_features = x
        if self.downsample:
            x = self.downsample(x)
        x = self.between_module(x, output_dict, previous_output_dict)
        if self.upsample:
            x = self.upsample(x)
        if self.add_skip_connection:
            x = torch.cat([x, encoded_features], axis=1)
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, output_dict, previous_output_dict)
        return x


class Bottleneck(nn.Module):
    def __init__(self, tensor_shape):
        super(Bottleneck, self).__init__()
        c, h, w = tensor_shape
        self.in_shape = tensor_shape
        self.out_shape = tensor_shape

        # Define RNN cell
        if config.refine_net_use_rnn:
            rnn_cells = []
            for i in range(config.refine_net_rnn_num_cells):
                if config.refine_net_rnn_type == 'CRNN':
                    rnn_cells.append(CRNNCell(input_size=config.refine_net_num_features,
                                              hidden_size=config.refine_net_num_features))
                elif config.refine_net_rnn_type == 'CLSTM':
                    rnn_cells.append(CLSTMCell(input_size=config.refine_net_num_features,
                                               hidden_size=config.refine_net_num_features))
                elif config.refine_net_rnn_type == 'CGRU':
                    rnn_cells.append(CGRUCell(input_size=config.refine_net_num_features,
                                              hidden_size=config.refine_net_num_features))
            self.rnn_cells = nn.ModuleList(rnn_cells)

    def forward(self, bottleneck_features, output_dict, previous_output_dict):
        if config.refine_net_use_rnn:
            for i, rnn_cell in enumerate(self.rnn_cells):
                suffix = '_%d' % i

                # Retrieve previous hidden/cell states if any
                previous_states = None
                if previous_output_dict is not None:
                    previous_states = previous_output_dict['refinenet_rnn_states' + suffix]

                # Inference through RNN cell
                states = rnn_cell(bottleneck_features, previous_states)

                # Decide what the output is and store back current states
                if isinstance(states, tuple):
                    rnn_features = states[0]
                    output_dict['refinenet_rnn_states' + suffix] = states
                else:
                    rnn_features = states
                    output_dict['refinenet_rnn_states' + suffix] = states
                    bottleneck_features = rnn_features

        return bottleneck_features


class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()

        in_c = 4 if config.load_screen_content else 1
        do_skip = config.refine_net_use_skip_connections

        # CNN backbone (ResNet-based)
        # - Replace first layer to take 5-channel input
        bottleneck = Bottleneck((config.refine_net_num_features, 5, 8))
        wrapped = WrapEncoderDecoder(in_shape=[256, 5, 8],
                                     out_shape=[256, 5, 8],
                                     module_to_wrap=bottleneck,
                                     num_encoder_blocks=2,
                                     add_skip_connection=do_skip)
        wrapped = WrapEncoderDecoder(in_shape=[128, 9, 16],
                                     out_shape=[128, 9, 16],
                                     module_to_wrap=wrapped,
                                     num_encoder_blocks=2,
                                     add_skip_connection=do_skip)
        wrapped = WrapEncoderDecoder(in_shape=[64, 18, 32],
                                     out_shape=[64, 18, 32],
                                     module_to_wrap=wrapped,
                                     num_encoder_blocks=2,
                                     add_skip_connection=do_skip)
        wrapped = WrapEncoderDecoder(in_shape=[32, 36, 64],
                                     out_shape=[32, 36, 64],
                                     module_to_wrap=wrapped,
                                     num_encoder_blocks=2,
                                     add_skip_connection=do_skip)
        wrapped = WrapEncoderDecoder(in_shape=[16, 72, 128],
                                     out_shape=[16, 72, 128],
                                     module_to_wrap=wrapped,
                                     add_skip_connection=do_skip)
        self.initial = nn.Sequential(
            nn.Conv2d(in_c, 16, kernel_size=3, padding=1),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
        )
        self.network = wrapped
        self.final = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Initializations
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.zeros_(self.final[-2].weight)

    def forward(self, input_dict, output_dict, previous_output_dict=None):
        # Form input image by concatenating (channel-wise) the screen frame and heatmap.
        input_heatmap = output_dict['heatmap_initial']
        scaled_heatmap = F.interpolate(
            input_heatmap, (config.screen_size[1], config.screen_size[0]),
            mode='bilinear', align_corners=False,
        )

        if config.load_screen_content:
            input_image = torch.cat([input_dict['screen_frame'], scaled_heatmap], axis=1)
        else:
            input_image = scaled_heatmap

        # Run through network
        input_features = self.initial(input_image)
        final_heatmap = self.final(
            self.network(input_features, output_dict, previous_output_dict)
        )
        output_dict['heatmap_final'] = final_heatmap
