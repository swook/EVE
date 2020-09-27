# Towards End-to-end Video-based Eye Tracking

The code accompanying our ECCV 2020 publication and dataset, EVE.

* Authors: [Seonwook Park](https://ait.ethz.ch/people/spark/), [Emre Aksan](https://ait.ethz.ch/people/eaksan/), [Xucong Zhang](https://ait.ethz.ch/people/zhang/), and [Otmar Hilliges](https://ait.ethz.ch/people/hilliges/)
* Project page: https://ait.ethz.ch/projects/2020/EVE/


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

TODO

#### Automatic logging to Google Sheets

TODO

### Running inference

TODO

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

A: We plan to make evaluations on the test set possible as soon as possible via a web service with a public leaderboard. In this way, we strive to make evaluations on our test set consistent and reliable, and encourage competition in the field of video-based gaze estimation.
