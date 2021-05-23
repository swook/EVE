# Towards End-to-end Video-based Eye Tracking

The code accompanying our ECCV 2020 publication and dataset, EVE.

* Authors: [Seonwook Park](https://ait.ethz.ch/people/spark/), [Emre Aksan](https://ait.ethz.ch/people/eaksan/), [Xucong Zhang](https://ait.ethz.ch/people/zhang/), and [Otmar Hilliges](https://ait.ethz.ch/people/hilliges/)
* Project page: https://ait.ethz.ch/projects/2020/EVE/
* Codalab (test set evaluation and public leaderboard): https://competitions.codalab.org/competitions/28954


## Setup

Preferably, setup a Docker image or virtual environment ([virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/install.html) is recommended) for this repository. Please note that we have tested this code-base in the following environments:
* Ubuntu 18.04 / A Linux-based cluster system (CentOS 7.8)
* Python 3.6 / Python 3.7
* PyTorch 1.5.1

Clone this repository somewhere with:

    git clone git@github.com:swook/EVE
    cd EVE/

Then from the base directory of this repository, install all dependencies with:

    pip install -r requirements.txt

Please note the [PyTorch official installation guide](https://pytorch.org/get-started/locally/) for setting up the `torch` and `torchvision` packages on your specific system.

You will also need to setup **ffmpeg** for video decoding. On Linux, we recommend installing distribution-specific packages (usually named `ffmpeg`). If necessary, check out the [official download page](https://ffmpeg.org/download.html) or [compilation instructions](https://trac.ffmpeg.org/wiki/CompilationGuide).


## Usage

### Information on the code framework

#### Configuration file system

All available configuration parameters are defined in `src/core/config_default.py`.

In order to override the default values, one can do:

1. Pass the parameter via a command-line parameter to `train.py` or `inference.py`. Note that in this case, replace all `_` characters with `-`. E.g. the config. parameter `refine_net_enabled` becomes `--refine-net-enabled 1`. Note that boolean parameters can be passed in via either `0/no/false` or `1/yes/true`.
2. Create a JSON file such as `src/configs/eye_net.json` or `src/configs/refine_net.json`.

The order of application are:
1. Default parameters
2. JSON-provided parameters, in order of JSON file declaration. For instance, in the command `python train.py config1.json config2.json`, `config2.json` overrides `config1.json` entries should there be any overlap.
3. CLI-provided parameters.

#### Automatic logging to Google Sheets

This framework implements an automatic logging code of all parameters, loss terms, and metrics to a Google Sheets document. This is done by the `gspread` library. To enable this possibility, follow these instructions:

1. Follow the instructions at https://gspread.readthedocs.io/en/latest/oauth2.html#for-end-users-using-oauth-client-id
2. Set `--gsheet-secrets-json-file` to a path to the credentials JSON file, and set `--gsheet-workbook-key` to the document key. This key is the part after `https://docs.google.com/spreadsheets/d/` and before any query or hash parameters.

An example config JSON file can be found at `src/configs/sample_gsheet.json`.

### Training a model

To train a model, simply run `python train.py` from `src/` with the appropriate configuration changes that are desired (see __"Configuration file system"__ above).

Note, that in order to resume the training of an existing model you must provide the path to the output folder via the `--resume-from` argument.

Also, at every fresh run of `train.py`, a unique identifier is generated to produce a unique output folder in `outputs/EVE/`. Hence, it is recommended to use the Google Sheets logging feature (see __"Automatic logging to Google Sheets"__) to keep track of your models.

### Running inference

The single-sample inference script at `src/inference.py` takes in the same arguments as `train.py` but expects two arguments in particular:

* `--input-path` is the path to a `basler.mp4` or `webcam_l.mp4` or `webcam_c.mp4` or `webcam_r.mp4` that exists in the EVE dataset.
* `--output-path` is a path to a desired output location (ending in `.mp4`).

This script works for both training, validation, and test samples and shows the reference point-of-gaze ground-truth when available.

## Citation
If using this code-base and/or the EVE dataset in your research, please cite the following publication:

    @inproceedings{Park2020ECCV,
      author    = {Seonwook Park and Emre Aksan and Xucong Zhang and Otmar Hilliges},
      title     = {Towards End-to-end Video-based Eye-Tracking},
      year      = {2020},
      booktitle = {European Conference on Computer Vision (ECCV)}
    }

## Q&A

**Q: How do I use this code for screen-based eye tracking?**

A: This code does not offer actual eye tracking. Rather, it concerns the benchmarking of the video-based gaze estimation methods outlined in the original paper. Extending this code to support an easy-to-use software for screen-based eye tracking is somewhat non-trivial, due to requirements on camera calibration (intrinsics, extrinsics), and an efficient pipeline for accurate and stable real-time eye or face patch extraction. Thus, we consider this to be beyond the scope of this code repository.

**Q: Where are the test set labels?**

A: Our public evaluation server and leaderboard are hosted by Codalab at https://competitions.codalab.org/competitions/28954. This allows for evaluations on our test set to be consistent and reliable, and encourage competition in the field of video-based gaze estimation. Please note that the performance reported by Codalab is not strictly speaking comparable to the original paper's results, as we only perform evaluation on a large subset of the full test set. We recommend acquiring the updated performance figures from the [leaderboard](https://competitions.codalab.org/competitions/28954#results).
