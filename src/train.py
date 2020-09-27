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

import torch
from torch.nn import functional as F
import torchvision.utils as vutils

import core.training as training
from datasources import EVESequences_train, EVESequences_val
from models.eve import EVE

logger = logging.getLogger(__name__)
config, device = training.script_init_common()

# Specify datasets used
train_dataset_paths = [
    ('eve_train', EVESequences_train, config.datasrc_eve, config.train_stimuli, config.train_cameras),  # noqa
]
validation_dataset_paths = [
    ('eve_val', EVESequences_val, config.datasrc_eve, config.test_stimuli, config.test_cameras),
]
train_data, test_data = training.init_datasets(train_dataset_paths, validation_dataset_paths)

# Define model
model = EVE()
print(model)
model = model.to(device)

# Optimizer
optimizers = [
    torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    ),
]

# Setup
model, optimizers, tensorboard = training.setup_common(model, optimizers)


# Training
for current_step, loss_terms, outputs, images_to_log_to_tensorboard \
        in training.main_loop_iterator(model, optimizers, train_data, test_data, tensorboard):

    # NOTE: If we fill in index i, it would pick the optimizer at index i.
    #       In this case, there is only one optimizer, and one variant of the full loss.
    loss_terms.append(outputs['full_loss'])

    if training.step_modulo(current_step, config.tensorboard_images_every_n_steps):
        if config.load_screen_content:
            # Screen content and gaze history
            screen = outputs['screen_frame'].detach()
            gaze_history = outputs['initial_gaze_history'].detach()
            gaze_history = F.interpolate(gaze_history,
                                         (config.screen_size[1], config.screen_size[0]),
                                         mode='bilinear', align_corners=False)
            gaze_history = torch.clamp(gaze_history, 0.0, 1.0)
            gaze_history = gaze_history.repeat(1, 3, 1, 1)
            screen_plus_history = torch.cat([
                screen,
                torch.mul(screen, gaze_history),
                gaze_history,
            ], axis=2)  # row-wise, so vconcat
            images_to_log_to_tensorboard['train/screen_plus_initial_history'] = \
                vutils.make_grid(screen_plus_history, normalize=True, scale_each=True)

            # Refined gaze history
            gaze_history = outputs['refined_gaze_history'].detach()
            gaze_history = F.interpolate(gaze_history,
                                         (config.screen_size[1], config.screen_size[0]),
                                         mode='bilinear', align_corners=False)
            gaze_history = torch.clamp(gaze_history, 0.0, 1.0)
            gaze_history = gaze_history.repeat(1, 3, 1, 1)
            screen_plus_history = torch.cat([
                screen,
                torch.mul(screen, gaze_history),
                gaze_history,
            ], axis=2)  # row-wise, so vconcat
            images_to_log_to_tensorboard['train/screen_plus_refined_history'] = \
                vutils.make_grid(screen_plus_history, normalize=True, scale_each=True)

            # Initial gaze heatmap
            initial_heatmap = outputs['initial_heatmap'].detach()
            images_to_log_to_tensorboard['train/1_initial_heatmap'] = \
                vutils.make_grid(initial_heatmap, normalize=True, scale_each=True)

            # Final gaze heatmap
            final_heatmap = outputs['final_heatmap'].detach()
            images_to_log_to_tensorboard['train/2_final_heatmap'] = \
                vutils.make_grid(final_heatmap, normalize=True, scale_each=True)

            # Groundtruth gaze heatmap
            gt_heatmap = outputs['gt_heatmap'].detach()
            images_to_log_to_tensorboard['train/0_gt_heatmap'] = \
                vutils.make_grid(gt_heatmap, normalize=True, scale_each=True)

# Do final test on full test sets (without subsampling)
# During training, live validation only occurs on a randomly selected sub-set.
training.do_final_full_test(model, test_data, tensorboard)

# Exit without hanging
training.cleanup_and_quit(train_data, test_data, tensorboard)
