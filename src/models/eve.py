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

from core import DefaultConfig
from losses.cross_entropy import CrossEntropyLoss
from losses.euclidean import EuclideanLoss
from losses.angular import AngularLoss
from losses.mse import MSELoss
from losses.l1 import L1Loss
from utils.load_model import load_weights_for_instance

from .common import (batch_make_heatmaps, batch_make_gaze_history_maps, soft_argmax,
                     to_screen_coordinates, apply_offset_augmentation,
                     calculate_combined_gaze_direction)
from .eye_net import EyeNet
from .refine_net import RefineNet

config = DefaultConfig()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cross_entropy_loss = CrossEntropyLoss()
euclidean_loss = EuclideanLoss()
angular_loss = AngularLoss()
mse_loss = MSELoss()
l1_loss = L1Loss()


class EVE(nn.Module):
    def __init__(self, output_predictions=False):
        super(EVE, self).__init__()
        self.output_predictions = output_predictions

        # Network to estimate gaze direction and pupil size (mm)
        self.eye_net = EyeNet()
        if config.eye_net_load_pretrained:
            load_weights_for_instance(self.eye_net)
        if config.eye_net_frozen:
            for param in self.eye_net.parameters():
                param.requires_grad = False

        # Network to refine estimated gaze based on:
        #   a) history of point of gaze (PoG)
        #   b) screen content
        self.refine_net = RefineNet() if config.refine_net_enabled else None
        if config.refine_net_enabled and config.refine_net_load_pretrained:
            load_weights_for_instance(self.refine_net)

    def forward(self, full_input_dict, create_images=False, current_epoch=None):
        if self.training:  # pick first source
            assert len(full_input_dict) == 1  # for now say there's 1 training data source
            full_input_dict = next(iter(full_input_dict.values()))

        # There are some labels that we need to calculate ourselves
        self.calculate_additional_labels(full_input_dict, current_epoch=current_epoch)

        # NOTE: In general, parts of the architecture will read from `input_dict`
        #       and write to `output_dict`. However, as we want some modularity,
        #       the routines will always look for previous-step estimates first,
        #       which will exist in `output_dict`.
        #
        #       For example, if the key `left_gaze_origin` exists in `output_dict`,
        #       the following routines will use this estimate instead of the
        #       ground-truth existing in `input_dict`.

        intermediate_dicts = []  # One entry per time step in sequence
        initial_heatmap_history = []
        refined_heatmap_history = []

        sequence_len = next(iter(full_input_dict.values())).shape[1]
        for t in range(sequence_len):

            # Create new dict formed only of the specific camera view's data
            sub_input_dict = {}
            for k, v in full_input_dict.items():
                if isinstance(v, torch.Tensor):
                    sub_v = v[:, t, :] if v.ndim > 2 else v[:, t]
                    sub_input_dict[k] = sub_v

            # Step 0) Define output structure that will hold
            #         - intermediate outputs
            #         - final outputs
            #         - individual loss terms
            sub_output_dict = {}

            # Step 1a) From each eye patch, estimate gaze direction and pupil size
            previous_output_dict = (intermediate_dicts[-1] if len(intermediate_dicts) > 0 else None)
            self.eye_net(sub_input_dict, sub_output_dict, side='left',
                         previous_output_dict=previous_output_dict)
            self.eye_net(sub_input_dict, sub_output_dict, side='right',
                         previous_output_dict=previous_output_dict)

            # During training: add random offsets to gaze directions
            if self.training and config.refine_net_do_offset_augmentation:
                self.from_g_to_PoG_history(full_input_dict, sub_input_dict, sub_output_dict,
                                           input_suffix='initial',
                                           output_suffix='initial_unaugmented',
                                           heatmap_history=None,
                                           gaze_heatmap_sigma=config.gaze_heatmap_sigma_initial,
                                           history_heatmap_sigma=config.gaze_heatmap_sigma_history)
                for side in ('left', 'right'):
                    sub_output_dict[side + '_g_initial_unaugmented'] = \
                            sub_output_dict[side + '_g_initial']
                    sub_output_dict[side + '_g_initial'] = apply_offset_augmentation(
                        sub_output_dict[side + '_g_initial'],
                        sub_input_dict['head_R'],
                        sub_input_dict[side + '_kappa_fake'],
                    )
                self.from_g_to_PoG_history(full_input_dict, sub_input_dict, sub_output_dict,
                                           input_suffix='initial',
                                           output_suffix='initial_augmented',
                                           heatmap_history=None,
                                           gaze_heatmap_sigma=config.gaze_heatmap_sigma_initial,
                                           history_heatmap_sigma=config.gaze_heatmap_sigma_history)

            # Step 1b) Estimate PoG, create heatmaps, and heatmap history
            self.from_g_to_PoG_history(full_input_dict, sub_input_dict, sub_output_dict,
                                       input_suffix='initial',
                                       output_suffix='initial',
                                       heatmap_history=initial_heatmap_history,
                                       gaze_heatmap_sigma=config.gaze_heatmap_sigma_initial,
                                       history_heatmap_sigma=config.gaze_heatmap_sigma_history)

            # Step 2) Digest screen content frame
            if self.refine_net:
                self.refine_net(sub_input_dict, sub_output_dict,
                                previous_output_dict=previous_output_dict)

                # Step 2b) Update refined heatmap history
                refined_heatmap_history.append(sub_output_dict['heatmap_final'])
                refined_gaze_history_maps = batch_make_gaze_history_maps(
                    full_input_dict['timestamps'], refined_heatmap_history,
                    full_input_dict['PoG_px_tobii_validity'],
                ) if 'PoG_px_tobii' in full_input_dict else None

                # Step 3) Yield refined final PoG estimate(s)
                sub_output_dict['PoG_px_final'] = soft_argmax(sub_output_dict['heatmap_final'])
                cm_per_px = 0.1 * sub_input_dict['millimeters_per_pixel']
                sub_output_dict['PoG_cm_final'] = torch.mul(
                    sub_output_dict['PoG_px_final'], cm_per_px,
                )

                # and gaze direction
                sub_output_dict['g_final'] = calculate_combined_gaze_direction(
                    sub_input_dict['o'],
                    10.0 * sub_output_dict['PoG_cm_final'],
                    sub_input_dict['left_R'],  # by definition, 'left_R' == 'right_R'
                    sub_input_dict['camera_transformation'],
                )

            # Store back outputs
            intermediate_dicts.append(sub_output_dict)

        # Merge intermediate outputs over time steps to yield BxTxF tensors
        full_intermediate_dict = {}
        for k in intermediate_dicts[0].keys():
            sample = intermediate_dicts[0][k]
            if not isinstance(sample, torch.Tensor):
                continue
            full_intermediate_dict[k] = torch.stack([
                intermediate_dicts[i][k] for i in range(sequence_len)
            ], axis=1)

        # Copy over some values that we want to yield as NN output
        output_dict = {}
        for k in full_intermediate_dict.keys():
            if k.startswith('output_'):
                output_dict[k] = full_intermediate_dict[k]

        output_dict['left_pupil_size'] = full_intermediate_dict['left_pupil_size']
        output_dict['right_pupil_size'] = full_intermediate_dict['right_pupil_size']

        if config.load_full_frame_for_visualization:
            # Copy over some values manually
            if 'left_g_tobii' in full_input_dict:
                output_dict['left_g_gt'] = full_input_dict['left_g_tobii']
                output_dict['PoG_px_gt'] = full_input_dict['PoG_px_tobii']
                output_dict['PoG_px_gt_validity'] = full_input_dict['PoG_px_tobii_validity']
            output_dict['left_g_initial'] = full_intermediate_dict['left_g_initial']
            output_dict['PoG_px_initial'] = full_intermediate_dict['PoG_px_initial']
            if config.refine_net_enabled:
                output_dict['g_final'] = full_intermediate_dict['g_final']
                output_dict['PoG_px_final'] = full_intermediate_dict['PoG_px_final']

        if self.output_predictions:
            output_dict['timestamps'] = full_input_dict['timestamps']
            output_dict['o'] = full_input_dict['o']
            output_dict['left_R'] = full_input_dict['left_R']
            output_dict['head_R'] = full_input_dict['head_R']
            output_dict['g_initial'] = full_intermediate_dict['g_initial']
            output_dict['PoG_px_initial'] = full_intermediate_dict['PoG_px_initial']
            output_dict['PoG_cm_initial'] = full_intermediate_dict['PoG_cm_initial']
            output_dict['millimeters_per_pixel'] = full_input_dict['millimeters_per_pixel']
            output_dict['pixels_per_millimeter'] = full_input_dict['pixels_per_millimeter']
            output_dict['camera_transformation'] = full_input_dict['camera_transformation']
            output_dict['inv_camera_transformation'] = full_input_dict['inv_camera_transformation']

            # Ground-truth related data
            if 'g' in full_input_dict:
                output_dict['g'] = full_input_dict['g']
                output_dict['validity'] = full_input_dict['PoG_px_tobii_validity']
                output_dict['PoG_cm'] = full_input_dict['PoG_cm_tobii']
                output_dict['PoG_px'] = full_input_dict['PoG_px_tobii']

            if self.refine_net:
                output_dict['g_final'] = full_intermediate_dict['g_final']
                output_dict['PoG_px_final'] = full_intermediate_dict['PoG_px_final']
                output_dict['PoG_cm_final'] = full_intermediate_dict['PoG_cm_final']

        # Calculate all loss terms and metrics (scores)
        self.calculate_losses_and_metrics(full_input_dict, full_intermediate_dict, output_dict)

        # Calculate the final combined (and weighted) loss
        full_loss = torch.zeros(()).to(device)

        # Add all losses for the eye network
        if 'loss_ang_left_g_initial' in output_dict:
            full_loss += config.loss_coeff_g_ang_initial * (
                output_dict['loss_ang_left_g_initial'] +
                output_dict['loss_ang_right_g_initial']
            )
        if 'loss_mse_left_PoG_cm_initial' in output_dict \
                and config.loss_coeff_PoG_cm_initial > 0.0:
            full_loss += config.loss_coeff_PoG_cm_initial * (
                output_dict['loss_mse_left_PoG_cm_initial'] +
                output_dict['loss_mse_right_PoG_cm_initial']
            )
        if 'loss_l1_left_pupil_size' in output_dict:
            full_loss += config.loss_coeff_pupil_size * (
                output_dict['loss_l1_left_pupil_size'] +
                output_dict['loss_l1_right_pupil_size']
            )

        # Add all losses for the GazeRefineNet
        if 'loss_mse_PoG_cm_final' in output_dict:
            full_loss += config.loss_coeff_PoG_cm_final * output_dict['loss_mse_PoG_cm_final']
        if 'loss_ce_heatmap_initial' in output_dict:
            full_loss += config.loss_coeff_heatmap_ce_initial * \
                    output_dict['loss_ce_heatmap_initial']
        if 'loss_ce_heatmap_final' in output_dict:
            full_loss += config.loss_coeff_heatmap_ce_final * output_dict['loss_ce_heatmap_final']
        if 'loss_mse_heatmap_final' in output_dict:
            full_loss += config.loss_coeff_heatmap_mse_final * output_dict['loss_mse_heatmap_final']

        output_dict['full_loss'] = full_loss

        # Store away tensors for visualization
        if create_images:
            if config.load_full_frame_for_visualization:
                output_dict['both_eye_patch'] = torch.cat([
                    full_input_dict['right_eye_patch'], full_input_dict['left_eye_patch'],
                ], axis=4)
            if config.load_screen_content:
                output_dict['screen_frame'] = full_input_dict['screen_frame'][:, -1, :]
            if 'history_initial' in full_intermediate_dict:
                output_dict['initial_gaze_history'] = full_intermediate_dict['history_initial'][:, -1, :]  # noqa
            if 'heatmap_initial' in full_intermediate_dict:
                output_dict['initial_heatmap'] = full_intermediate_dict['heatmap_initial'][:, -1, :]
            if 'heatmap_final' in full_intermediate_dict:
                output_dict['final_heatmap'] = full_intermediate_dict['heatmap_final'][:, -1, :]
                output_dict['refined_gaze_history'] = refined_gaze_history_maps
            if 'heatmap_final' in full_input_dict:
                output_dict['gt_heatmap'] = full_input_dict['heatmap_final'][:, -1, :]
        return output_dict

    def calculate_losses_and_metrics(self, input_dict, intermediate_dict, output_dict):
        # Initial estimates of gaze direction and PoG
        for side in ('left', 'right'):
            input_key = side + '_g_tobii'
            interm_key = (side + '_g_initial_unaugmented'
                          if self.training and config.refine_net_do_offset_augmentation
                          else side + '_g_initial')
            output_key = side + '_g_initial'
            if interm_key in intermediate_dict and input_key in input_dict:
                output_dict['loss_ang_' + output_key] = angular_loss(
                    intermediate_dict[interm_key], input_key, input_dict,
                )

            input_key = side + '_PoG_cm_tobii'
            interm_key = (side + '_PoG_cm_initial_unaugmented'
                          if self.training and config.refine_net_do_offset_augmentation
                          else side + '_PoG_cm_initial')
            output_key = side + '_PoG_cm_initial'
            if interm_key in intermediate_dict and input_key in input_dict:
                output_dict['loss_mse_' + output_key] = mse_loss(
                    intermediate_dict[interm_key], input_key, input_dict,
                )
                output_dict['metric_euc_' + output_key] = euclidean_loss(
                    intermediate_dict[interm_key], input_key, input_dict,
                )

            input_key = side + '_PoG_tobii'
            interm_key = side + '_PoG_px_initial'
            if interm_key in intermediate_dict and input_key in input_dict:
                output_dict['metric_euc_' + interm_key] = euclidean_loss(
                    intermediate_dict[interm_key], input_key, input_dict,
                )

            # Pupil size in mm
            input_key = side + '_p'
            interm_key = side + '_pupil_size'
            if interm_key in intermediate_dict and input_key in input_dict:
                output_dict['loss_l1_' + interm_key] = l1_loss(
                    intermediate_dict[interm_key], input_key, input_dict,
                )

        # Left-right consistency
        if 'left_PoG_tobii' in input_dict and 'right_PoG_tobii' in input_dict:
            intermediate_dict['right_PoG_cm_initial_validity'] = (
                input_dict['left_PoG_tobii_validity'] &
                input_dict['right_PoG_tobii_validity']
            )
            output_dict['loss_mse_lr_consistency'] = mse_loss(
                intermediate_dict['left_PoG_cm_initial'],
                'right_PoG_cm_initial', intermediate_dict,
            )
            output_dict['metric_euc_lr_consistency'] = euclidean_loss(
                intermediate_dict['left_PoG_cm_initial'],
                'right_PoG_cm_initial', intermediate_dict,
            )

        # Initial heatmap CE loss
        input_key = output_key = 'heatmap_initial'
        interm_key = ('heatmap_initial_unaugmented'
                      if self.training and config.refine_net_do_offset_augmentation
                      else 'heatmap_initial')
        if interm_key in intermediate_dict and input_key in input_dict:
            output_dict['loss_ce_' + output_key] = cross_entropy_loss(
                intermediate_dict[interm_key], input_key, input_dict,
            )

        # Refined heatmap MSE loss
        input_key = interm_key = 'heatmap_final'
        if interm_key in intermediate_dict and input_key in input_dict:
            output_dict['loss_ce_' + interm_key] = cross_entropy_loss(
                intermediate_dict[interm_key], input_key, input_dict,
            )
            output_dict['loss_mse_' + interm_key] = mse_loss(
                intermediate_dict[interm_key], input_key, input_dict,
            )

        # Metrics after applying kappa augmentation
        if config.refine_net_do_offset_augmentation:
            input_key = 'PoG_px_tobii'
            interm_key = 'PoG_px_initial_unaugmented'
            if interm_key in intermediate_dict and input_key in input_dict:
                output_dict['metric_euc_' + interm_key] = euclidean_loss(
                    intermediate_dict[interm_key], input_key, input_dict,
                )

            input_key = 'PoG_cm_tobii'
            interm_key = 'PoG_cm_initial_unaugmented'
            if interm_key in intermediate_dict and input_key in input_dict:
                output_dict['metric_euc_' + interm_key] = euclidean_loss(
                    intermediate_dict[interm_key], input_key, input_dict,
                )

            input_key = 'g'
            interm_key = 'g_initial_unaugmented'
            if interm_key in intermediate_dict and input_key in input_dict:
                output_dict['metric_ang_' + interm_key] = angular_loss(
                    intermediate_dict[interm_key], input_key, input_dict,
                )

        # Initial gaze
        input_key = 'PoG_px_tobii'
        interm_key = 'PoG_px_initial'
        if interm_key in intermediate_dict and input_key in input_dict:
            output_dict['loss_mse_' + interm_key] = mse_loss(
                intermediate_dict[interm_key], input_key, input_dict,
            )
            output_dict['metric_euc_' + interm_key] = euclidean_loss(
                intermediate_dict[interm_key], input_key, input_dict,
            )

        input_key = 'PoG_cm_tobii'
        interm_key = 'PoG_cm_initial'
        if interm_key in intermediate_dict and input_key in input_dict:
            output_dict['loss_mse_' + interm_key] = mse_loss(
                intermediate_dict[interm_key], input_key, input_dict,
            )
            output_dict['metric_euc_' + interm_key] = euclidean_loss(
                intermediate_dict[interm_key], input_key, input_dict,
            )

        input_key = 'g'
        interm_key = 'g_initial'
        if interm_key in intermediate_dict and input_key in input_dict:
            output_dict['metric_ang_' + interm_key] = angular_loss(
                intermediate_dict[interm_key], input_key, input_dict,
            )

        # Refine gaze
        input_key = 'PoG_px_tobii'
        interm_key = 'PoG_px_final'
        if interm_key in intermediate_dict and input_key in input_dict:
            output_dict['loss_mse_' + interm_key] = mse_loss(
                intermediate_dict[interm_key], input_key, input_dict,
            )
            output_dict['metric_euc_' + interm_key] = euclidean_loss(
                intermediate_dict[interm_key], input_key, input_dict,
            )

        input_key = 'PoG_cm_tobii'
        interm_key = 'PoG_cm_final'
        if interm_key in intermediate_dict and input_key in input_dict:
            output_dict['loss_mse_' + interm_key] = mse_loss(
                intermediate_dict[interm_key], input_key, input_dict,
            )
            output_dict['metric_euc_' + interm_key] = euclidean_loss(
                intermediate_dict[interm_key], input_key, input_dict,
            )

        input_key = 'g'
        interm_key = 'g_final'
        if interm_key in intermediate_dict and input_key in input_dict:
            output_dict['metric_ang_' + interm_key] = angular_loss(
                intermediate_dict[interm_key], input_key, input_dict,
            )

    def calculate_additional_labels(self, full_input_dict, current_epoch=None):
        sample_entry = next(iter(full_input_dict.values()))
        batch_size = sample_entry.shape[0]
        sequence_len = sample_entry.shape[1]

        # PoG in mm
        for side in ('left', 'right'):
            if (side + '_PoG_tobii') in full_input_dict:
                full_input_dict[side + '_PoG_cm_tobii'] = torch.mul(
                    full_input_dict[side + '_PoG_tobii'],
                    0.1 * full_input_dict['millimeters_per_pixel'],
                ).detach()
                full_input_dict[side + '_PoG_cm_tobii_validity'] = \
                    full_input_dict[side + '_PoG_tobii_validity']

        # Fake kappa to be used during training
        # Mirror the yaw angle to handle different eyes
        if self.training and config.refine_net_do_offset_augmentation:

            # Curriculum learning on kappa
            assert(current_epoch is not None)
            assert(isinstance(current_epoch, float))
            kappa_std = config.refine_net_offset_augmentation_sigma
            kappa_std = np.radians(kappa_std)

            # Create systematic noise
            # This is consistent throughout a given sequence
            left_kappas = np.random.normal(size=(batch_size, 2), loc=0.0, scale=kappa_std)
            right_kappas = np.random.normal(size=(batch_size, 2), loc=0.0, scale=kappa_std)
            kappas = {
                'left': np.repeat(np.expand_dims(left_kappas, axis=1), sequence_len, axis=1),
                'right': np.repeat(np.expand_dims(right_kappas, axis=1), sequence_len, axis=1),
            }

            for side in ('left', 'right'):
                # Store kappa
                kappas_side = kappas[side]
                kappa_tensor = torch.tensor(kappas_side.astype(np.float32)).to(device)
                full_input_dict[side + '_kappa_fake'] = kappa_tensor

        # 3D origin for L/R combined gaze
        if 'left_o' in full_input_dict:
            full_input_dict['o'] = torch.mean(torch.stack([
                full_input_dict['left_o'], full_input_dict['right_o'],
            ], axis=-1), axis=-1).detach()
            full_input_dict['o_validity'] = full_input_dict['left_o_validity']

        if 'left_PoG_tobii' in full_input_dict:
            # Average of left/right PoG values
            full_input_dict['PoG_px_tobii'] = torch.mean(torch.stack([
                full_input_dict['left_PoG_tobii'],
                full_input_dict['right_PoG_tobii'],
            ], axis=-1), axis=-1).detach()
            full_input_dict['PoG_cm_tobii'] = torch.mean(torch.stack([
                full_input_dict['left_PoG_cm_tobii'],
                full_input_dict['right_PoG_cm_tobii'],
            ], axis=-1), axis=-1).detach()
            full_input_dict['PoG_px_tobii_validity'] = (
                full_input_dict['left_PoG_tobii_validity'].bool() &
                full_input_dict['right_PoG_tobii_validity'].bool()
            ).detach()
            full_input_dict['PoG_cm_tobii_validity'] = full_input_dict['PoG_px_tobii_validity']

            if config.refine_net_enabled:
                # Heatmaps (both initial and final)
                # NOTE: input is B x T x F
                full_input_dict['heatmap_initial'] = torch.stack([
                    batch_make_heatmaps(full_input_dict['PoG_px_tobii'][b, :],
                                        config.gaze_heatmap_sigma_initial)
                    * full_input_dict['PoG_px_tobii_validity'][b, :].float().view(-1, 1, 1, 1)
                    for b in range(batch_size)
                ], axis=0).detach()
                full_input_dict['heatmap_history'] = torch.stack([
                    batch_make_heatmaps(full_input_dict['PoG_px_tobii'][b, :],
                                        config.gaze_heatmap_sigma_history)
                    * full_input_dict['PoG_px_tobii_validity'][b, :].float().view(-1, 1, 1, 1)
                    for b in range(batch_size)
                ], axis=0).detach()
                full_input_dict['heatmap_final'] = torch.stack([
                    batch_make_heatmaps(full_input_dict['PoG_px_tobii'][b, :],
                                        config.gaze_heatmap_sigma_final)
                    * full_input_dict['PoG_px_tobii_validity'][b, :].float().view(-1, 1, 1, 1)
                    for b in range(batch_size)
                ], axis=0).detach()
                full_input_dict['heatmap_initial_validity'] = \
                    full_input_dict['PoG_px_tobii_validity']
                full_input_dict['heatmap_history_validity'] = \
                    full_input_dict['PoG_px_tobii_validity']
                full_input_dict['heatmap_final_validity'] = \
                    full_input_dict['PoG_px_tobii_validity']

        # 3D gaze direction for L/R combined gaze
        if 'PoG_cm_tobii' in full_input_dict:
            full_input_dict['g'] = torch.stack([
                calculate_combined_gaze_direction(
                    full_input_dict['o'][b, :],
                    10.0 * full_input_dict['PoG_cm_tobii'][b, :],
                    full_input_dict['left_R'][b, :],
                    full_input_dict['camera_transformation'][b, :],
                )
                for b in range(batch_size)
            ], axis=0)
            full_input_dict['g_validity'] = full_input_dict['PoG_cm_tobii_validity']

    def from_g_to_PoG_history(self, full_input_dict, sub_input_dict, sub_output_dict,
                              input_suffix, output_suffix,
                              heatmap_history, gaze_heatmap_sigma, history_heatmap_sigma):

        # Handle case for GazeCapture and MPIIGaze
        if 'inv_camera_transformation' not in full_input_dict:
            return

        # Step 1) Calculate PoG from given gaze
        for side in ('left', 'right'):
            origin = (sub_output_dict[side + '_o']
                      if side + '_o' in sub_output_dict else sub_input_dict[side + '_o'])
            direction = sub_output_dict[side + '_g_' + input_suffix]
            rotation = (sub_output_dict[side + '_R']
                        if side + '_R' in sub_output_dict else sub_input_dict[side + '_R'])
            PoG_mm, PoG_px = to_screen_coordinates(origin, direction, rotation, sub_input_dict)
            sub_output_dict[side + '_PoG_cm_' + output_suffix] = 0.1 * PoG_mm
            sub_output_dict[side + '_PoG_px_' + output_suffix] = PoG_px

        # Step 1b) Calculate average PoG
        sub_output_dict['PoG_px_' + output_suffix] = torch.mean(torch.stack([
            sub_output_dict['left_PoG_px_' + output_suffix],
            sub_output_dict['right_PoG_px_' + output_suffix],
        ], axis=-1), axis=-1)
        sub_output_dict['PoG_cm_' + output_suffix] = torch.mean(torch.stack([
            sub_output_dict['left_PoG_cm_' + output_suffix],
            sub_output_dict['right_PoG_cm_' + output_suffix],
        ], axis=-1), axis=-1)
        sub_output_dict['PoG_mm_' + output_suffix] = \
            10.0 * sub_output_dict['PoG_cm_' + output_suffix]

        # Step 1b) Calculate the combined gaze (L/R)
        sub_output_dict['g_' + output_suffix] = \
            calculate_combined_gaze_direction(
                sub_input_dict['o'],
                sub_output_dict['PoG_mm_' + output_suffix],
                sub_input_dict['left_R'],  # by definition, 'left_R' == 'right_R'
                sub_input_dict['camera_transformation'],
            )

        if config.refine_net_enabled:
            # Step 2) Create heatmaps from PoG estimates
            sub_output_dict['heatmap_' + output_suffix] = \
                batch_make_heatmaps(sub_output_dict['PoG_px_' + output_suffix], gaze_heatmap_sigma)

            if heatmap_history is not None:
                # Step 3) Create gaze history maps
                heatmap_history.append(
                    batch_make_heatmaps(sub_output_dict['PoG_px_' + output_suffix],
                                        history_heatmap_sigma)
                )
                if 'PoG_px_tobii' in full_input_dict:
                    gaze_history_maps = batch_make_gaze_history_maps(
                        full_input_dict['timestamps'], heatmap_history,
                        full_input_dict['PoG_px_tobii_validity'],
                    )
                    sub_output_dict['history_' + output_suffix] = gaze_history_maps
