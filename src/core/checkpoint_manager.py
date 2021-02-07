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
import logging
import os
import shutil

import torch

from core import DefaultConfig

config = DefaultConfig()
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CheckpointManager(object):

    __model = None
    __optimizers = None
    __suffix = '.pt'

    def __init__(self, model, optimizers):
        self.__model = model
        self.__optimizers = optimizers

    def __save(self, ofdir):
        assert not os.path.isdir(ofdir)
        if hasattr(self.__model, 'module'):  # case where nn.DataParallel was used
            state_dict = self.__model.module.state_dict()
        else:
            state_dict = self.__model.state_dict()
        os.makedirs(ofdir)

        # Determine prefices
        prefices = set()
        for k in state_dict.keys():
            words = k.split('.')
            prefices.add(words[0])

        # Save each prefix into own file
        for prefix in prefices:
            sub_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith(prefix + '.'):
                    sub_state_dict[k] = v
            torch.save(sub_state_dict, '%s/%s%s' % (ofdir, prefix, self.__suffix))

        # Save each optimizer's state
        for i, optimizer in enumerate(self.__optimizers):
            output_path = '%s/optimizer_%d%s' % (ofdir, i, self.__suffix)
            torch.save(optimizer.state_dict(), output_path)

        logger.info('> Saved parameters to: %s' % ofdir)

    def __load(self, ifdir):
        assert os.path.isdir(ifdir)
        full_state_dict = {}

        # Gather state_dicts from directory
        ifpaths = [
            p for p in glob.glob(ifdir + '/*' + self.__suffix)
            if os.path.isfile(p) and not os.path.basename(p).startswith('optimizer_')
        ]
        for ifpath in ifpaths:
            sub_state_dict = torch.load(ifpath, map_location=device)
            for k, v in sub_state_dict.items():
                full_state_dict[k] = v
            logger.info('> Loaded model parameters from: %s' % ifpath)

        # Do the actual loading
        self.__model.load_state_dict(full_state_dict)

        # Load each optimizer's state
        optimizer_checkpoint_paths = [
            p for p in glob.glob(ifdir + '/optimizer_*' + self.__suffix)
            if os.path.isfile(p)
        ]
        for checkpoint_path in optimizer_checkpoint_paths:
            optimizer_index = int(os.path.basename(checkpoint_path).split('.')[0].split('_')[-1])
            if optimizer_index < len(self.__optimizers):
                self.__optimizers[optimizer_index].load_state_dict(
                    torch.load(checkpoint_path, map_location=device)
                )
                logger.info('> Loaded optimizer parameters from: %s' % checkpoint_path)

        step = int(os.path.split(ifdir)[-1][:-3])
        return step

    def __output_dir(self):
        return os.path.relpath(os.path.join(
            self.__model.output_dir,
            'checkpoints',
        ))

    def __output_fpath(self, current_step):
        return os.path.relpath(os.path.join(
            self.__output_dir(),
            ('%07d' % current_step) + self.__suffix,
        ))

    def save_at_step(self, current_step):
        self.__save(self.__output_fpath(current_step))
        self.__only_keep_n_checkpoints()

    def __get_available_checkpoints(self):
        output_dir = self.__output_dir()
        return sorted([
            (int(os.path.split(fn)[-1].split('.')[0]), fn)
            for fn in glob.glob(os.path.join(output_dir, '*' + self.__suffix))
            if fn.endswith(self.__suffix) and os.path.isdir(fn)
        ])

    def __only_keep_n_checkpoints(self):
        available = self.__get_available_checkpoints()
        if len(available) > config.checkpoints_keep_n:
            for step, fpath in available[:-config.checkpoints_keep_n]:
                shutil.rmtree(fpath)
                logger.info('> Removing parameters folder at: %s' % fpath)

    def load_last_checkpoint(self):
        return self.__load_last_checkpoint()

    def __load_last_checkpoint(self):
        available = self.__get_available_checkpoints()
        if len(available) > 0:
            return self.__load(available[-1][1])
        else:
            return 0
