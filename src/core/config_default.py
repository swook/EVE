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
import glob
import json
import os
import sys
import zipfile

import logging
logger = logging.getLogger(__name__)


class DefaultConfig(object):

    # Tag to separate Emre and my experiments
    identifier_suffix = ''

    # Misc. notes
    note = ''

    # Data sources
    datasrc_eve = '/path/to/eve/dataset'

    # Data loading
    video_decoder_codec = 'libx264'  # libx264 | nvdec
    assumed_frame_rate = 10  # We will skip frames from source videos accordingly
    max_sequence_len = 30  # In frames assuming 10Hz
    face_size = [256, 256]  # width, height
    eyes_size = [128, 128]  # width, height
    screen_size = [128, 72]  # width, height
    actual_screen_size = [1920, 1080]  # DO NOT CHANGE
    camera_frame_type = 'eyes'  # full | face | eyes
    load_screen_content = False
    load_full_frame_for_visualization = False

    train_cameras = ['basler', 'webcam_l', 'webcam_c', 'webcam_r']
    train_stimuli = ['image', 'video', 'wikipedia']
    test_cameras = ['basler', 'webcam_l', 'webcam_c', 'webcam_r']
    test_stimuli = ['image', 'video', 'wikipedia']

    # Inference
    input_path = ''
    output_path = ''

    # Training
    skip_training = False
    fully_reproducible = False  # enable with possible penalty of performance

    batch_size = 16
    weight_decay = 0.001
    num_epochs = 10.0

    train_data_workers = 8

    log_every_n_steps = 1  # NOTE: Every other interval has to be a multiple of this!!!
    tensorboard_scalars_every_n_steps = 1
    tensorboard_images_every_n_steps = 10
    tensorboard_learning_rate_every_n_steps = 100

    # Learning rate
    base_learning_rate = 0.0005

    @property
    def learning_rate(self):
        return self.batch_size * self.base_learning_rate
    # Available strategies:
    #     'exponential': step function with exponential decay
    #     'cyclic':      spiky down-up-downs (with exponential decay of peaks)
    num_warmup_epochs = 0.0  # No. of epochs to warmup LR from base to target
    lr_decay_strategy = 'none'
    lr_decay_factor = 0.5
    lr_decay_epoch_interval = 0.5

    # Gradient Clipping
    do_gradient_clipping = True
    gradient_clip_by = 'norm'  # 'norm' or 'value'
    gradient_clip_amount = 5.0

    # Eye gaze network configuration
    eye_net_load_pretrained = False
    eye_net_frozen = False
    eye_net_use_rnn = True
    eye_net_rnn_type = 'GRU'  # 'RNN' | 'LSTM' | 'GRU'
    eye_net_rnn_num_cells = 1
    eye_net_rnn_num_features = 128
    eye_net_static_num_features = 128
    eye_net_use_head_pose_input = True
    loss_coeff_PoG_cm_initial = 0.0
    loss_coeff_g_ang_initial = 1.0
    loss_coeff_pupil_size = 1.0

    # Conditional refine network configuration
    refine_net_enabled = False
    refine_net_load_pretrained = False

    refine_net_do_offset_augmentation = True
    refine_net_offset_augmentation_sigma = 3.0

    refine_net_use_skip_connections = True

    refine_net_use_rnn = True
    refine_net_rnn_type = 'CGRU'  # 'CRNN' | 'CLSTM' | 'CGRU'
    refine_net_rnn_num_cells = 1
    refine_net_num_features = 64
    loss_coeff_heatmap_ce_initial = 0.0
    loss_coeff_heatmap_ce_final = 1.0
    loss_coeff_heatmap_mse_final = 0.0
    loss_coeff_PoG_cm_final = 0.001

    # Heatmaps
    gaze_heatmap_size = [128, 72]
    gaze_heatmap_sigma_initial = 10.0  # in pixels
    gaze_heatmap_sigma_history = 3.0  # in pixels
    gaze_heatmap_sigma_final = 5.0  # in pixels
    gaze_history_map_decay_per_ms = 0.999

    # Evaluation
    test_num_samples = 128
    test_batch_size = 128
    test_data_workers = 0
    test_every_n_steps = 500
    full_test_batch_size = 128
    full_test_data_workers = 4

    codalab_eval_batch_size = 128
    codalab_eval_data_workers = 1

    # Checkpoints management
    checkpoints_save_every_n_steps = 100
    checkpoints_keep_n = 3
    resume_from = ''

    # Google Sheets related
    gsheet_secrets_json_file = ''
    gsheet_workbook_key = ''

    # Below lie necessary methods for working configuration tracking

    __instance = None

    # Make this a singleton class
    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__filecontents = cls.__get_config_file_contents()
            cls.__pycontents = cls.__get_python_file_contents()
            cls.__immutable = True
        return cls.__instance

    def import_json(self, json_path, strict=True):
        """Import JSON config to over-write existing config entries."""
        assert os.path.isfile(json_path)
        assert not hasattr(self.__class__, '__imported_json_path')
        logger.info('Loading ' + json_path)
        with open(json_path, 'r') as f:
            json_string = f.read()
        self.import_dict(json.loads(json_string), strict=strict)
        self.__class__.__imported_json_path = json_path
        self.__class__.__filecontents[os.path.basename(json_path)] = json_string

    def override(self, key, value):
        self.__class__.__immutable = False
        setattr(self, key, value)
        self.__class__.__immutable = True

    def import_dict(self, dictionary, strict=True):
        """Import a set of key-value pairs from a dict to over-write existing config entries."""
        self.__class__.__immutable = False
        for key, value in dictionary.items():
            if strict is True:
                if not hasattr(self, key):
                    raise ValueError('Unknown configuration key: ' + key)
                if type(getattr(self, key)) is float and type(value) is int:
                    value = float(value)
                else:
                    assert type(getattr(self, key)) is type(value)
                if not isinstance(getattr(DefaultConfig, key), property):
                    setattr(self, key, value)
            else:
                if hasattr(DefaultConfig, key):
                    if not isinstance(getattr(DefaultConfig, key), property):
                        setattr(self, key, value)
                else:
                    setattr(self, key, value)
        self.__class__.__immutable = True

    def __get_config_file_contents():
        """Retrieve and cache default and user config file contents."""
        out = {}
        for relpath in ['config_default.py']:
            path = os.path.relpath(os.path.dirname(__file__) + '/' + relpath)
            assert os.path.isfile(path)
            with open(path, 'r') as f:
                out[os.path.basename(path)] = f.read()
        return out

    def __get_python_file_contents():
        """Retrieve and cache default and user config file contents."""
        out = {}
        base_path = os.path.relpath(os.path.dirname(__file__) + '/../')
        source_fpaths = [
            p for p in glob.glob(base_path + '/**/*.py')
            if not p.startswith('./3rdparty/')
        ]
        source_fpaths += [os.path.relpath(sys.argv[0])]
        for fpath in source_fpaths:
            assert os.path.isfile(fpath)
            with open(fpath, 'r') as f:
                out[fpath[2:]] = f.read()
        return out

    def get_all_key_values(self):
        return dict([
            (key, getattr(self, key))
            for key in dir(self)
            if not key.startswith('_DefaultConfig')
            and not key.startswith('__')
            and not callable(getattr(self, key))
        ])

    def get_full_json(self):
        return json.dumps(self.get_all_key_values(), indent=4)

    def write_file_contents(self, target_base_dir):
        """Write cached config file contents to target directory."""
        assert os.path.isdir(target_base_dir)

        # Write config file contents
        target_dir = target_base_dir + '/configs'
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
        outputs = {  # Also output flattened config
            'combined.json': self.get_full_json(),
        }
        outputs.update(self.__class__.__filecontents)
        for fname, content in outputs.items():
            fpath = os.path.relpath(target_dir + '/' + fname)
            with open(fpath, 'w') as f:
                f.write(content)
                logger.info('Written %s' % fpath)

        # Copy source folder contents over
        target_path = os.path.relpath(target_base_dir + '/src.zip')
        source_path = os.path.relpath(os.path.dirname(__file__) + '/../')
        filter_ = lambda x: x.endswith('.py') or x.endswith('.json')  # noqa
        with zipfile.ZipFile(target_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for root, dirs, files in os.walk(source_path):
                for file_or_dir in files + dirs:
                    full_path = os.path.join(root, file_or_dir)
                    if os.path.isfile(full_path) and filter_(full_path):
                        zip_file.write(
                            os.path.join(root, file_or_dir),
                            os.path.relpath(os.path.join(root, file_or_dir),
                                            os.path.join(source_path, os.path.pardir)))
        logger.info('Written source folder to %s' % os.path.relpath(target_path))

    def __setattr__(self, name, value):
        """Initial configs should not be overwritten!"""
        if self.__class__.__immutable:
            raise AttributeError('DefaultConfig instance attributes are immutable.')
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        """Initial configs should not be removed!"""
        if self.__class__.__immutable:
            raise AttributeError('DefaultConfig instance attributes are immutable.')
        else:
            super().__delattr__(name)
