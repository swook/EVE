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

import torch.utils.model_zoo

from core import DefaultConfig

config = DefaultConfig()
logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_url_pre = 'https://github.com/swook/EVE/releases/download/v0.0/'


def load_weights_for_instance(model_instance):
    from models.eye_net import EyeNet
    from models.refine_net import RefineNet
    if isinstance(model_instance, EyeNet):
        model_fname = 'eve_eyenet_'
        model_fname += config.eye_net_rnn_type if config.eye_net_use_rnn else 'static'
        model_fname += '.pt'
    elif isinstance(model_instance, RefineNet):
        model_fname = 'eve_refinenet_'
        model_fname += config.refine_net_rnn_type if config.refine_net_use_rnn else 'static'
        model_fname += '_oa' if config.refine_net_do_offset_augmentation else ''
        model_fname += '_skip' if config.refine_net_use_skip_connections else ''
        model_fname += '.pt'
    else:
        raise ValueError('Cannot load weights for given model instance: %s' %
                         model_instance.__class__)

    model_url = model_url_pre + model_fname

    # Load the weights
    state_dict_from_url = torch.utils.model_zoo.load_url(model_url, map_location=device)
    model_instance.load_state_dict(state_dict_from_url)
    logger.info('Loaded model weights from: %s' % model_url)
