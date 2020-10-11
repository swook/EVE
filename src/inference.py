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
import os

import cv2 as cv
import ffmpeg
import numpy as np
import torch

from core.config_default import DefaultConfig
import core.inference as inference
from models.eve import EVE

# Default singleton config object
config = DefaultConfig()

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Setup logger
logger = logging.getLogger(__name__)

# Run some routines from the training setup pipeline
inference.script_init_common()

# Initialize dataset and dataloader
dataset, dataloader = inference.init_dataset()

# Define and set up model
model = EVE(output_predictions=True).to(device)
model = inference.model_setup(model)

# Prepare output dir.
output_dir = os.path.dirname(config.output_path)
if output_dir > '':
    os.makedirs(output_dir, exist_ok=True)
video_handle = None

# Do inference
for step, inputs, outputs in inference.iterator(model, dataloader, create_images=True):

    # Visualize
    we_have_gt = 'left_g_gt' in outputs
    all_g_init = outputs['left_g_initial']
    if config.load_full_frame_for_visualization:
        all_eyes = outputs['both_eye_patch']
        all_eyes = (all_eyes + 1.0) * (255.0 / 2.0)
        all_eyes = all_eyes.astype(np.uint8)
        all_eyes = np.transpose(all_eyes, [0, 1, 3, 4, 2])[:, :, :, :, ::-1]
        if 'screen_full_frame' in inputs:
            all_screen = inputs['screen_full_frame']
            all_screen = all_screen.astype(np.uint8)
            all_screen = all_screen[:, :, :, :, ::-1]  # RGB to BGR
    all_PoG_init = outputs['PoG_px_initial']
    if config.load_screen_content:
        all_g_final = outputs['g_final']
        all_PoG_final = outputs['PoG_px_final']
    if we_have_gt:
        all_g_gt = outputs['left_g_gt']
        all_PoG_gt = outputs['PoG_px_gt']
        all_PoG_gt_validity = outputs['PoG_px_gt_validity']
    num_entries = all_g_init.shape[0]
    sequence_len = all_g_init.shape[1]
    for index in range(num_entries):
        if config.load_full_frame_for_visualization:
            eyes = all_eyes[index, :]
            if 'screen_full_frame' in inputs:
                screen = all_screen[index, :]
        g_init = all_g_init[index, :]
        PoG_init = all_PoG_init[index, :]
        if config.load_screen_content:
            g_final = all_g_final[index, :]
            PoG_final = all_PoG_final[index, :]
        if we_have_gt:
            g_gt = all_g_gt[index, :]
            PoG_gt = all_PoG_gt[index, :]
            gt_validity = all_PoG_gt_validity[index, :]
        final_out_frames = {
            # 'mirrored_eye_y': [eyes[t, :, 128:, :] for t in range(sequence_len)],
            # 'mirrored_eye_r': [eyes[t, :, 128:, :] for t in range(sequence_len)],
            # 'mirrored_eye_g': [eyes[t, :, 128:, :] for t in range(sequence_len)],
            # 'eye_y': [eyes[t, :, 128:, :] for t in range(sequence_len)],
            # 'eye_r': [eyes[t, :, 128:, :] for t in range(sequence_len)],
            # 'eye_g': [eyes[t, :, 128:, :] for t in range(sequence_len)],
            # 'eye_rg': [eyes[t, :, 128:, :] for t in range(sequence_len)],
            # 'eye_ry': [eyes[t, :, 128:, :] for t in range(sequence_len)],
            # 'mirrored_screen_r': [screen[t, :] for t in range(sequence_len)],
            # 'mirrored_screen_y': [screen[t, :] for t in range(sequence_len)],
            # 'mirrored_screen_g': [screen[t, :] for t in range(sequence_len)],
            # 'screen_r': [screen[t, :] for t in range(sequence_len)],
            # 'screen_y': [screen[t, :] for t in range(sequence_len)],
            # 'screen_g': [screen[t, :] for t in range(sequence_len)],
            # 'screen_yr': [screen[t, :] for t in range(sequence_len)],
            'screen_yrg': [screen[t, :] for t in range(sequence_len)],
        }

        for suffix, frames in final_out_frames.items():
            frames = [np.copy(frame) for frame in frames]

            # Choose whether to mirror frame
            _g_init = np.copy(g_init)
            _PoG_init = np.copy(PoG_init)
            if config.load_screen_content:
                _g_final = np.copy(g_final)
                _PoG_final = np.copy(PoG_final)
            if we_have_gt:
                _g_gt = np.copy(g_gt)
                _PoG_gt = np.copy(PoG_gt)
                _gt_validity = np.copy(gt_validity)
            if suffix.startswith('mirrored_'):
                frames = [np.ascontiguousarray(np.fliplr(frame)) for frame in frames]
                if config.load_screen_content:
                    _PoG_final[:, 0] = 1920.0 - _PoG_final[:, 0]
                    _g_final[:, 1] = -_g_final[:, 1]
                _PoG_init[:, 0] = 1920.0 - _PoG_init[:, 0]
                _g_init[:, 1] = -_g_init[:, 1]
                if we_have_gt:
                    _PoG_gt[:, 0] = 1920.0 - _PoG_gt[:, 0]
                    _g_gt[:, 1] = -_g_gt[:, 1]
            _all_valid = np.ones((sequence_len, ), dtype=np.bool)

            if 'screen' in suffix:
                # Choose what to draw
                to_draw = []
                last_bit = suffix.split('_')[-1]
                is_drawing_gt = 'r' in last_bit
                for char in list(last_bit):
                    if char == 'y':
                        to_draw.append(('Initial Estimate', _PoG_init, _all_valid,
                                       [0, 180, 180]))
                    elif char == 'g':
                        to_draw.append(('After Refinement (Ours)', _PoG_final, _all_valid,
                                       [0, 180, 0]))
                    elif char == 'r':
                        if we_have_gt:
                            to_draw.append(('Tobii Data (Groundtruth)', _PoG_gt, _gt_validity,
                                           [0, 0, 180]))
                    else:
                        raise ValueError('Invalid thing to draw: %s' % char)

                # Inset eye image
                if not suffix.startswith('mirrored_'):
                    for t in range(sequence_len):
                        eyes = cv.resize(all_eyes[index, t, :], (256, 128))
                        eh, ew, _ = eyes.shape
                        frames[t][-eh:, -ew:, :] = np.fliplr(eyes)

                if we_have_gt and is_drawing_gt:
                    # Draw error/residual "labels"
                    for label, PoG_list, validity, colour in to_draw:
                        for t, (x, y) in enumerate(PoG_list):
                            if 'Groundtruth' not in label and _gt_validity[t] == 1:
                                x_gt, y_gt = _PoG_gt[t, :]
                                cv.line(frames[t], (x, y), (x_gt, y_gt), color=[0, 0, 0],
                                        thickness=5, lineType=cv.LINE_AA)
                                cv.line(frames[t], (x, y), (x_gt, y_gt), color=colour,
                                        thickness=2, lineType=cv.LINE_AA)

                # Draw fixation circles
                for _, PoG_list, validity, colour in to_draw:
                    for t, (x, y) in enumerate(PoG_list):
                        if validity[t] == 1:
                            cv.circle(frames[t], (x, y), radius=14, color=[0, 0, 0],
                                      thickness=-1, lineType=cv.LINE_AA)
                            cv.circle(frames[t], (x, y), radius=10, color=colour,
                                      thickness=-1, lineType=cv.LINE_AA)

                # Now label with a "legend"
                for t in range(sequence_len):
                    offset_dy = 0
                    for label, _, _, colour in to_draw:
                        offset_x = 50
                        offset_y = 90 + offset_dy
                        cv.putText(frames[t], label, org=(offset_x, offset_y),
                                   fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=1.6,
                                   color=[0, 0, 0], thickness=9, lineType=cv.LINE_AA)
                        cv.putText(frames[t], label, org=(offset_x, offset_y),
                                   fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=1.6,
                                   color=colour, thickness=2, lineType=cv.LINE_AA)
                        offset_dy += 80

            elif 'eye' in suffix:
                ow, oh = 512, 512
                frames = [cv.resize(frame, (ow, oh)) for frame in frames]

                # Choose what to draw
                to_draw = []
                last_bit = suffix.split('_')[-1]
                for char in list(last_bit):
                    if char == 'y':
                        to_draw.append((_g_init, _all_valid, [0, 180, 180]))
                    elif char == 'r':
                        if we_have_gt:
                            to_draw.append((_g_gt, _gt_validity, [0, 0, 180]))
                    elif char == 'g':
                        to_draw.append((_g_final, _all_valid, [0, 180, 0]))
                    else:
                        raise ValueError('Invalid thing to draw: %s' % char)

                # Draw the rays
                for g_list, validity, colour in to_draw:
                    for t, (pitch, yaw) in enumerate(g_list):
                        if validity[t] == 0:
                            continue
                        length = 200.0
                        dx = -length * np.cos(pitch) * np.sin(yaw)
                        dy = -length * np.sin(pitch)
                        hw, hh = int(ow / 2), int(oh / 2)
                        cv.arrowedLine(frames[t], (hw, hh),
                                       tuple(np.round([hw + dx, hh + dy]).astype(int)),
                                       [0, 0, 0], thickness=10, line_type=cv.LINE_AA,
                                       tipLength=0.2)
                        cv.arrowedLine(frames[t], (hw, hh),
                                       tuple(np.round([hw + dx, hh + dy]).astype(int)),
                                       colour, thickness=4, line_type=cv.LINE_AA,
                                       tipLength=0.2)

            # Now write video file
            oh, ow, _ = frames[0].shape
            if video_handle is None:
                video_handle = (
                    ffmpeg
                    .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='%dx%d' % (ow, oh),
                           framerate=10)
                    .output(config.output_path, pix_fmt='yuv420p', r=10, loglevel='quiet')
                    .overwrite_output()
                    .run_async(pipe_stdin=True, quiet=True)
                )
            for frame in frames:
                video_handle.stdin.write(frame.astype(np.uint8).tobytes())

# We are done now, let's close the output file
video_handle.stdin.close()
video_handle.wait()
print('> Wrote %s' % config.output_path)
