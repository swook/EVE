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
import argparse
from collections import OrderedDict
import functools
import gc
import hashlib
import logging
import os
import sys
import time

import coloredlogs
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from core import DefaultConfig, CheckpointManager, GoogleSheetLogger, Tensorboard

config = DefaultConfig()

# Setup logger
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _convert_cli_arg_type(key, value):
    config_type = type(getattr(config, key))
    if config_type == bool:
        if value.lower() in ('true', 'yes', 'y') or value == '1':
            return True
        elif value.lower() in ('false', 'no', 'n') or value == '0':
            return False
        else:
            raise ValueError('Invalid input for bool config "%s": %s' % (key, value))
    else:
        return config_type(value)


def script_init_common():
    parser = argparse.ArgumentParser(description='Train a gaze estimation model.')
    parser.add_argument('-v', type=str, help='Desired logging level.', default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])
    parser.add_argument('config_json', type=str, nargs='*',
                        help=('Path to config in JSON format. '
                              'Multiple configs will be parsed in the specified order.'))
    for key in dir(config):
        if key.startswith('_DefaultConfig') or key.startswith('__'):
            continue
        if key in vars(DefaultConfig) and isinstance(vars(DefaultConfig)[key], property):
            continue
        value = getattr(config, key)
        value_type = type(value)
        arg_type = value_type
        if value_type == bool:
            # Handle booleans separately, otherwise arbitrary values become `True`
            arg_type = str
        if callable(value):
            continue
        parser.add_argument('--' + key.replace('_', '-'), type=arg_type, metavar=value,
                            help='Expected type is `%s`.' % value_type.__name__)
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set logger format and verbosity level
    coloredlogs.install(
        datefmt='%d/%m %H:%M:%S',
        fmt='%(asctime)s %(levelname)s %(message)s',
        level=args.v.upper(),
    )

    # Parse configs in order specified by user
    for json_path in args.config_json:
        config.import_json(json_path)

    # Apply configs passed through command line
    config.import_dict({
        key.replace('-', '_'): _convert_cli_arg_type(key, value)
        for key, value in vars(args).items()
        if value is not None and hasattr(config, key)
    })

    # Improve reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    if config.fully_reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    return config, device


def init_datasets(train_specs, test_specs):

    # Initialize training datasets
    train_data = OrderedDict()
    for tag, dataset_class, path, stimuli, cameras in train_specs:
        dataset = dataset_class(path,
                                cameras_to_use=cameras,
                                types_of_stimuli=stimuli)
        dataset.original_full_dataset = dataset
        dataloader = DataLoader(dataset,
                                batch_size=config.batch_size,
                                shuffle=True,
                                drop_last=True,
                                num_workers=config.train_data_workers,
                                pin_memory=True,
                                )
        train_data[tag] = {
            'dataset': dataset,
            'dataloader': dataloader,
        }
        logger.info('> Ready to use training dataset: %s' % tag)
        logger.info('          with number of videos: %d' % len(dataset))

    # Initialize test datasets
    test_data = OrderedDict()
    for tag, dataset_class, path, stimuli, cameras in test_specs:
        # Get the full dataset
        dataset = dataset_class(path,
                                cameras_to_use=cameras,
                                types_of_stimuli=stimuli,
                                live_validation=True)
        dataset.original_full_dataset = dataset
        # then subsample datasets for quicker testing
        num_subset = config.test_num_samples
        if len(dataset) > num_subset:
            subset = Subset(dataset, sorted(np.random.permutation(len(dataset))[:num_subset]))
            subset.original_full_dataset = dataset
            dataset = subset
        dataloader = DataLoader(dataset,
                                batch_size=config.test_batch_size,
                                shuffle=False,
                                num_workers=config.test_data_workers,
                                pin_memory=True,
                                )
        test_data[tag] = {
            'dataset': dataset,
            'dataset_class': dataset_class,
            'dataset_path': path,
            'dataloader': dataloader,
        }
        logger.info('> Ready to use evaluation dataset: %s' % tag)
        logger.info('           with number of entries: %d' % len(dataset.original_full_dataset))
        if dataset.original_full_dataset != dataset:
            logger.info('     of which we evaluate on just: %d' % len(dataset))

    return train_data, test_data


def setup_common(model, optimizers):
    identifier = (model.__class__.__name__ +
                  config.identifier_suffix + '/' +
                  time.strftime('%y%m%d_%H%M%S') + '.' +
                  hashlib.md5(config.get_full_json().encode('utf-8')).hexdigest()[:6]
                  )

    if len(config.resume_from) > 0:
        identifier = '/'.join(config.resume_from.split('/')[-2:])
        output_dir = config.resume_from

    else:
        output_dir = '../outputs/' + identifier

    # Initialize tensorboard
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    tensorboard = Tensorboard(output_dir)

    # Write source code to output dir
    # NOTE: do not over-write if resuming from an output directory
    if len(config.resume_from) == 0:
        config.write_file_contents(output_dir)

    # Log messages to file
    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(output_dir + '/messages.log')
    file_handler.setFormatter(root_logger.handlers[0].formatter)
    for handler in root_logger.handlers[1:]:  # all except stdout
        root_logger.removeHandler(handler)
    root_logger.addHandler(file_handler)

    # Print model details
    num_params = sum([
        np.prod(p.size())
        for p in filter(lambda p: p.requires_grad, model.parameters())
    ])
    logger.info('\nThere are %d trainable parameters.\n' % num_params)

    # Cache base and target learning rate for each optimizer
    for optimizer in optimizers:
        optimizer.target_lr = optimizer.param_groups[0]['lr']
        optimizer.base_lr = optimizer.target_lr / config.batch_size

    # Sneak in some extra information into the model class instance
    model.identifier = identifier
    model.output_dir = output_dir
    model.checkpoint_manager = CheckpointManager(model, optimizers)
    model.gsheet_logger = GoogleSheetLogger(model)
    model.last_epoch = 0.0
    model.last_step = 0

    # Load pre-trained model weights if available
    if len(config.resume_from) > 0:
        model.last_step = model.checkpoint_manager.load_last_checkpoint()

    return model, optimizers, tensorboard


def salvage_memory():
    """Try to free whatever memory that can be freed."""
    torch.cuda.empty_cache()
    gc.collect()


def get_training_batches(train_data_dicts):
    """Get training batches of data from all training data sources."""
    out = {}
    for tag, data_dict in train_data_dicts.items():
        if 'data_iterator' not in data_dict:
            data_dict['data_iterator'] = iter(data_dict['dataloader'])
        # Try to get data
        while True:
            try:
                out[tag] = next(data_dict['data_iterator'])
                break
            except StopIteration:
                del data_dict['data_iterator']
                salvage_memory()
                data_dict['data_iterator'] = iter(data_dict['dataloader'])

        # Move tensors to GPU
        for k, v in out[tag].items():
            if isinstance(v, torch.Tensor):
                out[tag][k] = v.detach()
                if k != 'screen_full_frame':
                    out[tag][k] = out[tag][k].to(device, non_blocking=True)
            else:
                out[tag][k] = v
    return out


def test_model_on_all(model, test_data_dicts, current_step, tensorboard=None,
                      log_key_prefix='test'):
    """Get training batches of data from all training data sources."""
    model.eval()
    salvage_memory()
    final_out = {}
    for tag, data_dict in test_data_dicts.items():
        with torch.no_grad():
            num_entries = len(data_dict['dataset'])
            for i, input_data in enumerate(data_dict['dataloader']):
                batch_size = next(iter(input_data.values())).shape[0]

                # Move tensors to GPU
                for k, v in input_data.items():
                    if isinstance(v, torch.Tensor):
                        input_data[k] = v.detach().to(device, non_blocking=True)

                # Inference
                batch_out = model(input_data, create_images=(i == 0))
                weighted_batch_out = dict([
                    (k, v.detach().cpu().numpy() * (batch_size / num_entries))
                    for k, v in batch_out.items() if v.dim() == 0
                ])
                if tag not in final_out:
                    final_out[tag] = dict([(k, 0.0) for k in weighted_batch_out.keys()])
                for k, v in weighted_batch_out.items():
                    final_out[tag][k] += v

                # Log images
                if i == 0:
                    assert tensorboard
                    if 'images' in batch_out:
                        import torchvision.utils as vutils
                        tensorboard.add_image(
                            log_key_prefix + '_%s/images' % tag,
                            vutils.make_grid(batch_out['images'].detach()[:8, :],
                                             nrow=1,  # One entry per row
                                             padding=20,
                                             normalize=True,
                                             scale_each=True,
                                             )
                        )

        # Calculate mean error over whole dataset
        logger.info('%10s test: %s' % ('[%s]' % tag,
                                       ', '.join(['%s: %.4g' % (k, final_out[tag][k])
                                                  for k in sorted(final_out[tag].keys())])))

        # Write to tensorboard
        if tensorboard:
            tensorboard.update_current_step(current_step)
            for k, v in final_out[tag].items():
                tensorboard.add_scalar(log_key_prefix + '_%s/%s' % (tag, k), v)

    # Log training metrics to Google Sheets
    for_gsheet = None
    if model.gsheet_logger.ready:
        for_gsheet = {}
        for tag, out in final_out.items():
            for k, v in out.items():
                for_gsheet[log_key_prefix + '/%s/%s' % (tag, k)] = v

    # Free up memory
    salvage_memory()

    return final_out, for_gsheet


def do_final_full_test(model, test_data, tensorboard):
    previously_registered_dataset_classes = {}
    for k, v in test_data.items():
        # Get the full dataset
        if 'dataloader' in test_data[k]:
            del v['dataloader']
        full_original_dataset = v['dataset'].original_full_dataset
        previously_registered_dataset_classes[k] = v['dataset']
        new_dataset = v['dataset_class'](
            v['dataset_path'],
            cameras_to_use=full_original_dataset.cameras_to_use,
            types_of_stimuli=full_original_dataset.types_of_stimuli,
            is_final_test=True,
        )
        test_data[k]['dataset'] = new_dataset
        test_data[k]['dataloader'] = DataLoader(new_dataset,
                                                batch_size=config.full_test_batch_size,
                                                shuffle=False,
                                                num_workers=config.full_test_data_workers,
                                                pin_memory=True,
                                                )
        logger.info('> Ready to do full test on dataset: %s' % k)
        logger.info('          with number of sequences: %d' % len(new_dataset))

    logger.info('# Now beginning full test on all evaluation sets.')
    logger.info('# Hold on tight, this might take a while.')
    logger.info('#')
    _, for_gsheet = test_model_on_all(model, test_data, model.last_step + 2,
                                      tensorboard=tensorboard,
                                      log_key_prefix='full_test')

    # Restore dataset class
    for k, v in test_data.items():
        test_data[k]['dataset'] = previously_registered_dataset_classes[k]

    # Clean up dataloaders
    for k, v in test_data.items():
        del v['dataloader']

    # Log training metrics to Google Sheets
    if for_gsheet is not None:
        model.gsheet_logger.update_or_append_row(for_gsheet)

    # Free memory
    salvage_memory()


def learning_rate_schedule(optimizer, epoch_len, tensorboard_log_func, step):
    num_warmup_steps = int(epoch_len * config.num_warmup_epochs)
    selected_lr = None
    if step < num_warmup_steps:
        b = optimizer.base_lr
        a = (optimizer.target_lr - b) / float(num_warmup_steps)
        selected_lr = a * step + b
    else:
        # Decay learning rate with step function and exponential decrease?
        new_step = step - num_warmup_steps
        epoch = new_step / float(epoch_len)
        current_interval = int(epoch / config.lr_decay_epoch_interval)
        if config.lr_decay_strategy == 'exponential':
            # Step function decay
            selected_lr = optimizer.target_lr * np.power(config.lr_decay_factor, current_interval)
        elif config.lr_decay_strategy == 'cyclic':
            # Note, we start from the up state (due to previous warmup stage)
            # so each period consists of down-up (not up-down)
            peak_a = optimizer.target_lr * np.power(config.lr_decay_factor, current_interval)
            peak_b = peak_a * config.lr_decay_factor
            half_interval = 0.5 * config.lr_decay_epoch_interval
            current_interval_start = current_interval * config.lr_decay_epoch_interval
            current_interval_half = current_interval_start + half_interval
            if epoch < current_interval_half:
                # negative slope (down from peak_a)
                slope = -(peak_a - optimizer.base_lr) / half_interval
            else:
                # positive slope (up to peak_b)
                slope = (peak_b - optimizer.base_lr) / half_interval
            selected_lr = slope * (epoch - current_interval_half) + optimizer.base_lr
        else:
            selected_lr = optimizer.target_lr

    # Log to Tensorboard and return
    if step_modulo(step, config.tensorboard_learning_rate_every_n_steps):
        tensorboard_log_func(selected_lr)
    return selected_lr


def step_modulo(current, interval_size):
    return current % interval_size == (interval_size - 1)


def main_loop_iterator(model, optimizers, train_data, test_data, tensorboard=None,
                       do_before_forward_pass=None):
    # Skip this entirely if requested
    if config.skip_training:
        return

    assert tensorboard is not None  # We assume this exists in LR schedule logging
    initial_step = model.last_step  # Allow resuming
    max_dataset_len = np.amax([len(data_dict['dataset']) for data_dict in train_data.values()])
    num_steps_per_epoch = int(max_dataset_len / config.batch_size)
    num_training_steps = int(config.num_epochs * num_steps_per_epoch)
    lr_schedulers = [
        torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            functools.partial(learning_rate_schedule, optimizer, num_steps_per_epoch,
                              functools.partial(tensorboard.add_scalar, 'lr/optim_%d' % i)),
        ) for i, optimizer in enumerate(optimizers)
    ]
    model.train()
    current_step = 0
    for current_step in range(initial_step, num_training_steps):
        current_epoch = (current_step * config.batch_size) / max_dataset_len  # fractional value
        tensorboard.update_current_step(current_step + 1)
        input_data = get_training_batches(train_data)

        # Set correct states before training iteration
        model.train()
        for optimizer in optimizers:
            optimizer.zero_grad()

        # If routine defined for before forward pass, do it
        if do_before_forward_pass:
            do_before_forward_pass(current_step)

        # Prepare keyword arguments to model inference
        forward_kwargs = {
            'create_images': step_modulo(
                current_step, config.tensorboard_images_every_n_steps
            ),
            'current_epoch': current_epoch,
        }

        # Forward pass and yield
        loss_terms = []
        images_to_log_to_tensorboard = {}
        outputs = model(input_data, **forward_kwargs)
        yield current_step, loss_terms, outputs, images_to_log_to_tensorboard

        # There should be as many loss terms as there are optimizers!
        assert len(loss_terms) == len(optimizers)

        # Prune out None values
        valid_loss_terms = []
        valid_optimizers = []
        for loss_term, optimizer in zip(loss_terms, optimizers):
            if loss_term is not None:
                valid_loss_terms.append(loss_term)
                valid_optimizers.append(optimizer)

        # Perform gradient calculations for each loss term
        for i, (loss, optimizer) in enumerate(zip(valid_loss_terms, valid_optimizers)):
            not_last = i < (len(optimizers) - 1)
            if not isinstance(loss, torch.Tensor):
                continue
            loss.backward(retain_graph=not_last)

        # Maybe clip gradients
        if config.do_gradient_clipping:
            if config.gradient_clip_by == 'norm':
                clip_func = nn.utils.clip_grad_norm_
            elif config.gradient_clip_by == 'value':
                clip_func = nn.utils.clip_grad_value_
            clip_amount = config.gradient_clip_amount
            clip_func(model.parameters(), clip_amount)

        # Apply gradients
        for optimizer in valid_optimizers:
            optimizer.step()

        # Print outputs
        if step_modulo(current_step, config.log_every_n_steps):
            metrics = dict([(k, v.detach().cpu().numpy())
                            for k, v in outputs.items()
                            if v.dim() == 0])
            for i, loss in enumerate(loss_terms):  # Add loss terms
                if loss is not None:
                    metrics['loss_%d' % (i + 1)] = loss.detach().cpu().numpy()

            log = ('Step %d, Epoch %.2f> ' % (current_step + 1, current_epoch)
                   + ', '.join(['%s: %.4g' % (k, metrics[k]) for k in sorted(metrics.keys())]))
            logger.info(log)

            # Log to Tensorboard
            if step_modulo(current_step, config.tensorboard_scalars_every_n_steps):
                for key, metric in metrics.items():
                    if key.startswith('loss_'):
                        key = key[len('loss_'):]
                        tensorboard.add_scalar('train_losses/%s' % key, metric)
                    elif key.startswith('metric_'):
                        key = key[len('metric_'):]
                        tensorboard.add_scalar('train_metrics/%s' % key, metric)
                    else:
                        tensorboard.add_scalar('train/%s' % key, metric)

                tensorboard.add_scalar('lr/epoch', current_epoch)

                if step_modulo(current_step, config.tensorboard_images_every_n_steps):
                    for k, img in images_to_log_to_tensorboard.items():
                        tensorboard.add_image(k, img)

            # Quit if NaNs
            there_are_NaNs = False
            for k, v in metrics.items():
                if np.any(np.isnan(v)):
                    logger.error('NaN encountered during training at value: %s' % k)
                    there_are_NaNs = True
            if there_are_NaNs:
                cleanup_and_quit(train_data, test_data, tensorboard)

        # We're done with the previous outputs
        del input_data, outputs, loss_terms, images_to_log_to_tensorboard

        # Save checkpoint
        if step_modulo(current_step, config.checkpoints_save_every_n_steps):
            model.checkpoint_manager.save_at_step(current_step + 1)

        # Full test over all evaluation datasets
        if step_modulo(current_step, config.test_every_n_steps):

            # Do test on subset of validation datasets
            _, for_gsheet = test_model_on_all(model, test_data, current_step + 1,
                                              tensorboard=tensorboard)

            # Log training metrics to Google Sheets
            if for_gsheet is not None:
                for_gsheet['Step'] = current_step + 1
                for_gsheet['Epoch'] = current_epoch
                for k, v in metrics.items():
                    for_gsheet['train/' + k] = v
                model.gsheet_logger.update_or_append_row(for_gsheet)

            # Free memory
            salvage_memory()

        # Remember what the last step/epoch were
        model.last_epoch = current_epoch
        model.last_step = current_step

        # Update learning rate
        # NOTE: should be last
        tensorboard.update_current_step(current_step + 2)
        for lr_scheduler in lr_schedulers:
            lr_scheduler.step(current_step + 1)

    # We're out of the training loop now, make a checkpoint
    current_step += 1
    model.checkpoint_manager.save_at_step(current_step + 1)

    # Close all dataloaders
    for k, v in list(train_data.items()) + list(test_data.items()):
        if 'data_iterator' in v:
            v['data_iterator'].__del__()
            del v['data_iterator']
        v['dataloader']
        del v['dataloader']

    # Clear memory where possible
    salvage_memory()


def eval_loop_iterator(model, dataset, dataloader, create_images=False):
    """Iterate through and evaluate for a dataset."""
    model.eval()
    salvage_memory()
    with torch.no_grad():
        # num_entries = len(dataset)
        for current_step, input_data in enumerate(dataloader):
            # batch_size = next(iter(input_data.values())).shape[0]

            # Move tensors to GPU
            input_data_gpu = {}
            for k, v in input_data.items():
                if isinstance(v, torch.Tensor):
                    input_data_gpu[k] = v.detach().to(device, non_blocking=True)

            forward_kwargs = {
                'create_images': create_images,
            }

            # Forward pass and yield
            outputs = model(input_data_gpu, **forward_kwargs)
            yield current_step, input_data, outputs

    # Free up memory
    salvage_memory()


def cleanup_and_quit(train_data, test_data, tensorboard):
    # Close tensorboard
    if tensorboard:
        tensorboard.__del__()

    # Close all dataloaders and datasets
    for k, v in list(train_data.items()) + list(test_data.items()):
        if 'data_iterator' in v:
            v['data_iterator'].__del__()
        # if 'dataset' in v:
        #     v['dataset'].original_full_dataset.__del__()
        for item in ['data_iterator', 'dataloader', 'dataset']:
            if item in v:
                del v[item]

    # Finally exit
    sys.exit(0)
