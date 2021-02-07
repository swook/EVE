"""Copyright 2021 ETH Zurich, Seonwook Park

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
import os

import torch

from core import DefaultConfig, CheckpointManager
import core.training as training
from datasources import EVESequences_test

# Default singleton config object
config = DefaultConfig()

# Setup logger
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def script_init_common():
    # Set inference-specific overrides
    config.override('fully_reproducible', True)
    config.override('refine_net_enabled', True)
    config.override('load_screen_content', True)
    config.override('load_full_frame_for_visualization', False)

    # Run the remaining routines from the training mode
    training.script_init_common()


def init_dataset():

    # Initialize dataset and dataloader
    dataset = EVESequences_test(
        config.datasrc_eve,
        is_final_test=True,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.codalab_eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.codalab_eval_data_workers,
        pin_memory=True,
    )

    return dataset, dataloader


def model_setup(model):
    assert len(config.resume_from) > 0
    assert os.path.isdir(config.resume_from)

    # Load pre-trained model weights
    model.checkpoint_manager = CheckpointManager(model, [])
    model.output_dir = config.resume_from
    model.last_step = model.checkpoint_manager.load_last_checkpoint()
    assert model.last_step > 0

    return model


def iterator(model, dataloader, **kwargs):
    model.eval()
    with torch.no_grad():
        for current_step, input_data in enumerate(dataloader):

            # Move tensors to device
            input_data_gpu = {}
            for k, v in input_data.items():
                if isinstance(v, torch.Tensor):
                    input_data_gpu[k] = v.detach().to(device, non_blocking=True)

            # Forward pass and yield
            outputs = model(input_data_gpu, **kwargs)

            # Convert data
            inputs_np = {
                k: v.numpy() if isinstance(v, torch.Tensor) else v
                for k, v in input_data.items()
            }
            outputs_np = {
                k: v.detach().cpu().numpy()
                for k, v in outputs.items()
                if isinstance(v, torch.Tensor)
            }
            yield current_step, inputs_np, outputs_np
