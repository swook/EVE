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
import gzip
import logging
import os
import pickle
import time
import zipfile

import numpy as np
import torch

from core.config_default import DefaultConfig
import core.eval_codalab as eval_codalab
from models.eve import EVE

# Default singleton config object
config = DefaultConfig()

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Setup logger
logger = logging.getLogger(__name__)

# Run some routines from the training setup pipeline
eval_codalab.script_init_common()

# Initialize dataset and dataloader
dataset, dataloader = eval_codalab.init_dataset()

# Define and set up model
model = EVE(output_predictions=True).to(device)
model = eval_codalab.model_setup(model)

# Do eval_codalab
processed_so_far = set()
outputs_to_write = {}
for step, inputs, outputs in eval_codalab.iterator(model, dataloader):

    batch_size = next(iter(outputs.values())).shape[0]

    for i in range(batch_size):
        participant = inputs['participant'][i]
        subfolder = inputs['subfolder'][i]
        camera = inputs['camera'][i]

        # Ensure that the sub-dicts exist.
        if participant not in outputs_to_write:
            outputs_to_write[participant] = {}
        if subfolder not in outputs_to_write[participant]:
            outputs_to_write[participant][subfolder] = {}

        # Store back to output structure
        keys_to_store = [
            'timestamps',
            'left_pupil_size',
            'right_pupil_size',
            'PoG_px_initial',
            'PoG_px_final',
        ]
        sub_dict = outputs_to_write[participant][subfolder]
        if camera in sub_dict:
            for key in keys_to_store:
                sub_dict[camera][key] = np.concatenate([sub_dict[camera][key],
                                                        outputs[key][i, :]], axis=0)
        else:
            sub_dict[camera] = {}
            for key in keys_to_store:
                sub_dict[camera][key] = outputs[key][i, :]

        sequence_key = (participant, subfolder, camera)
        if sequence_key not in processed_so_far:
            print('Handling %s/%s/%s' % sequence_key)
            processed_so_far.add(sequence_key)

# Write output file
output_fname = 'for_codalab_%s.pkl.gz' % time.strftime('%y%m%d_%H%M%S')
final_output_path = os.path.join(model.output_dir, output_fname)
with gzip.open(final_output_path, 'wb') as f:
    pickle.dump(outputs_to_write, f, protocol=3)

# Write output zip
zip_output_path = final_output_path.replace('.pkl.gz', '.zip')
with zipfile.ZipFile(zip_output_path, 'w') as zf:
    zf.write(final_output_path, arcname=output_fname)
