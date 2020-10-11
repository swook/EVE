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
import pickle
from typing import List

import cv2 as cv
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from core import DefaultConfig
from .common import predefined_splits, stimulus_type_from_folder_name, VideoReader

config = DefaultConfig()
logger = logging.getLogger(__name__)

source_to_fps = {
    'screen': 30,
    'basler': 60,
    'webcam_l': 30,
    'webcam_c': 30,
    'webcam_r': 30,
}

source_to_interval_ms = dict([
    (source, 1e3 / fps) for source, fps in source_to_fps.items()
])

sequence_segmentations = None
cache_pkl_path = './eve_segmentation_cache.pkl'


class EVESequencesBase(Dataset):

    def __init__(self, dataset_path: str,
                 participants_to_use: List[str] = None,
                 cameras_to_use: List[str] = None,
                 types_of_stimuli: List[str] = None,
                 stimulus_name_includes: str = '',
                 live_validation: bool = False,
                 is_final_test: bool = False):
        if types_of_stimuli is None:
            types_of_stimuli = ['image', 'video', 'wikipedia']
        if cameras_to_use is None:
            cameras_to_use = ['basler', 'webcam_l', 'webcam_c', 'webcam_r']
        assert('points' not in types_of_stimuli)  # NOTE: deal with this in another way

        self.path = dataset_path
        self.types_of_stimuli = types_of_stimuli
        self.stimulus_name_includes = stimulus_name_includes
        self.participants_to_use = participants_to_use
        self.cameras_to_use = cameras_to_use
        self.live_validation = live_validation
        self.is_final_test = is_final_test
        self.validation_data_cache = {}

        # Some sanity checks
        assert(len(self.participants_to_use) > 0)
        assert(30 > config.assumed_frame_rate)
        assert(30 % config.assumed_frame_rate == 0)

        # Load or calculate sequence segmentations (start/end indices)
        global cache_pkl_path, sequence_segmentations
        cache_pkl_path = (
            './segmentation_cache/%dHz_seqlen%d.pkl' % (
                config.assumed_frame_rate, config.max_sequence_len,
            )
        )
        if sequence_segmentations is None:
            if not os.path.isfile(cache_pkl_path):
                self.build_segmentation_cache()
                assert(os.path.isfile(cache_pkl_path))

            with open(cache_pkl_path, 'rb') as f:
                sequence_segmentations = pickle.load(f)

        # Register entries
        self.select_sequences()
        logger.info('Initialized dataset class for: %s' % self.path)

    def build_segmentation_cache(self):
        """Create support data structure for knowing how to segment (cut up) time sequences."""
        all_folders = sorted([
            d for d in os.listdir(self.path) if os.path.isdir(self.path + '/' + d)
        ])
        output_to_cache = {}
        for folder_name in all_folders:
            participant_path = '%s/%s' % (self.path, folder_name)
            assert(os.path.isdir(participant_path))
            output_to_cache[folder_name] = {}

            subfolders = sorted([
                p for p in os.listdir(participant_path)
                if os.path.isdir(os.path.join(participant_path, p))
                and p.split('/')[-1].startswith('step')
                and 'eye_tracker_calibration' not in p
            ])
            for subfolder in subfolders:
                subfolder_path = '%s/%s' % (participant_path, subfolder)
                output_to_cache[folder_name][subfolder] = {}

                # NOTE: We assume that the videos are synchronized and have the same length in time.
                #       This should be the case for the publicly released EVE dataset.
                for source in ('screen', 'basler', 'webcam_l', 'webcam_c', 'webcam_r'):
                    current_outputs = []
                    source_path_pre = '%s/%s' % (subfolder_path, source)
                    available_indices = np.loadtxt('%s.timestamps.txt' % source_path_pre)
                    num_available_indices = len(available_indices)

                    # Determine desired length and skips
                    fps = source_to_fps[source]
                    target_len_in_s = config.max_sequence_len / config.assumed_frame_rate
                    num_original_indices_in_sequence = fps * target_len_in_s
                    assert(num_original_indices_in_sequence.is_integer())
                    num_original_indices_in_sequence = int(
                        num_original_indices_in_sequence
                    )
                    index_interval = int(fps / config.assumed_frame_rate)
                    start_index = 0
                    while start_index < num_available_indices:
                        end_index = min(
                            start_index + num_original_indices_in_sequence,
                            num_available_indices
                        )
                        picked_indices = list(range(start_index, end_index, index_interval))
                        current_outputs.append(picked_indices)

                        # Move along sequence
                        start_index += num_original_indices_in_sequence

                    # Store back indices
                    if len(current_outputs) > 0:
                        output_to_cache[folder_name][subfolder][source] = current_outputs
                        # print('%s: %d' % (source_path_pre, len(current_outputs)))

        # Do the caching
        with open(cache_pkl_path, 'wb') as f:
            pickle.dump(output_to_cache, f)

        logger.info('> Stored indices of sequences to: %s' % cache_pkl_path)

    def select_sequences(self):
        """Select sequences (start/end indices) for the selected participants/cameras/stimuli."""
        self.all_subfolders = []
        for participant_name, participant_data in sequence_segmentations.items():
            if participant_name not in self.participants_to_use:
                continue

            for stimulus_name, stimulus_segments in participant_data.items():
                current_stimulus_type = stimulus_type_from_folder_name(stimulus_name)
                if current_stimulus_type not in self.types_of_stimuli:
                    continue
                if len(self.stimulus_name_includes) > 0:
                    if self.stimulus_name_includes not in stimulus_name:
                        continue

                for camera, all_indices in stimulus_segments.items():
                    if camera not in self.cameras_to_use:
                        continue

                    for i, indices in enumerate(all_indices):
                        self.all_subfolders.append({
                            'camera_name': camera,
                            'participant': participant_name,
                            'subfolder': stimulus_name,
                            'partial_path': '%s/%s' % (participant_name, stimulus_name),
                            'full_path': '%s/%s/%s' % (self.path, participant_name, stimulus_name),
                            'indices': indices,
                            'screen_indices': stimulus_segments['screen'][i],
                        })

    def __len__(self):
        return len(self.all_subfolders)

    def preprocess_frames(self, frames):
        # Expected input:  N x H x W x C
        # Expected output: N x C x H x W
        frames = np.transpose(frames, [0, 3, 1, 2])
        frames = frames.astype(np.float32)
        frames *= 2.0 / 255.0
        frames -= 1.0
        return frames

    def preprocess_screen_frames(self, frames):
        # Expected input:  N x H x W x C
        # Expected output: N x C x H x W
        frames = np.transpose(frames, [0, 3, 1, 2])
        frames = frames.astype(np.float32)
        frames *= 1.0 / 255.0
        return frames

    screen_frames_cache = {}

    def load_all_from_source(self, path, source, selected_indices):
        assert(source in ('basler', 'webcam_l', 'webcam_c', 'webcam_r', 'screen'))

        # Read HDF
        subentry = {}  # to output
        if source != 'screen':
            with h5py.File('%s/%s.h5' % (path, source), 'r') as hdf:
                for k1, v1 in hdf.items():
                    if isinstance(v1, h5py.Group):
                        subentry[k1] = np.copy(v1['data'][selected_indices])
                        subentry[k1 + '_validity'] = np.copy(v1['validity'][selected_indices])
                    else:
                        shape = v1.shape
                        subentry[k1] = np.repeat(np.reshape(v1, (1, *shape)),
                                                 repeats=config.max_sequence_len, axis=0)

            # Compute rotation matrices from rvec values
            subentry['head_R'] = np.stack([cv.Rodrigues(rvec)[0] for rvec in subentry['head_rvec']])

        if config.load_full_frame_for_visualization and source == 'screen':
            _, full_frames = VideoReader(path + '/' + source + '.mp4',
                                         frame_indices=selected_indices,
                                         is_async=False).get_frames()
            subentry['full_frame'] = full_frames

        # Get frames
        video_path = '%s/%s' % (path, source)
        output_size = None
        if source == 'screen':
            video_path += '.128x72.mp4'
            output_size = config.screen_size

        else:
            if config.camera_frame_type == 'full':
                video_path += '.mp4'
            elif config.camera_frame_type == 'face':
                video_path += '_face.mp4'
                output_size = (config.face_size[0], config.face_size[1])
            elif config.camera_frame_type == 'eyes':
                video_path += '_eyes.mp4'
                output_size = (2*config.eyes_size[0], config.eyes_size[1])
            else:
                raise ValueError('Unknown camera frame type: %s' % config.camera_frame_type)

        if self.is_final_test:
            if path not in self.screen_frames_cache:
                self.screen_frames_cache = {path: {}}
            if source not in self.screen_frames_cache[path]:
                self.screen_frames_cache[path][source] = \
                    VideoReader(video_path, is_async=False,
                                output_size=output_size).get_frames()
            timestamps, frames = self.screen_frames_cache[path][source]
            timestamps = timestamps[selected_indices]
            frames = frames[selected_indices]
        else:
            timestamps, frames = VideoReader(video_path, frame_indices=selected_indices,
                                             is_async=False, output_size=output_size).get_frames()

        # Collect and return
        subentry['timestamps'] = np.asarray(timestamps, dtype=np.int)
        frames = (
            self.preprocess_screen_frames(frames)
            if source == 'screen' else
            self.preprocess_frames(frames)
        )
        if source == 'screen':
            subentry['frame'] = frames
        else:
            ew, eh = config.eyes_size
            subentry['left_eye_patch'] = frames[:, :, :, ew:]
            subentry['right_eye_patch'] = frames[:, :, :, :ew]

        # Pad as necessary with zero value and zero validity
        for key, value in subentry.items():
            if value.shape[0] < config.max_sequence_len:
                pad_len = config.max_sequence_len - value.shape[0]
                if pad_len > 0:
                    subentry[key] = np.pad(
                        value,
                        pad_width=[(0, pad_len if i == 0 else 0) for i in range(value.ndim)],
                        mode='constant',
                        constant_values=(False if value.dtype is np.bool else 0.0),
                    )

        return subentry

    def __getitem__(self, idx):
        # Retrieve sub-folder specification
        spec = self.all_subfolders[idx]
        path = spec['full_path']
        source = spec['camera_name']
        indices = spec['indices']
        screen_indices = spec['screen_indices']

        # Check cache if requested
        # NOTE: this only works with num_workers=0 as otherwise memory is not shared nor persisted.
        if self.live_validation:
            cache_key = '%s/%s' % (path, source)
            if cache_key in self.validation_data_cache:
                return self.validation_data_cache[cache_key]

        # Grab all data
        entry = self.load_all_from_source(path, source, indices)
        if config.load_screen_content:
            sub_entry = self.load_all_from_source(path, 'screen', screen_indices)
            for k, v in sub_entry.items():  # Add to full output dict
                entry['screen_%s' % k] = v

        # Add meta data
        entry['participant'] = spec['participant']
        entry['subfolder'] = spec['subfolder']
        entry['camera'] = spec['camera_name']

        torch_entry = dict([
            (k, torch.from_numpy(a)) if isinstance(a, np.ndarray) else (k, a)
            for k, a in entry.items()
        ])

        if self.live_validation:
            self.validation_data_cache[cache_key] = torch_entry
        return torch_entry


class EVESequences_train(EVESequencesBase):
    def __init__(self, dataset_path: str, **kwargs):
        super(EVESequences_train, self).__init__(
            dataset_path,
            participants_to_use=predefined_splits['train'],
            **kwargs,
        )


class EVESequences_val(EVESequencesBase):
    def __init__(self, dataset_path: str, **kwargs):
        super(EVESequences_val, self).__init__(
            dataset_path,
            participants_to_use=predefined_splits['val'],
            **kwargs,
        )


class EVESequences_test(EVESequencesBase):
    def __init__(self, dataset_path: str, **kwargs):
        super(EVESequences_test, self).__init__(
            dataset_path,
            participants_to_use=predefined_splits['test'],
            **kwargs,
        )
