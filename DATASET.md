# Documentation regarding the EVE dataset

## Introduction

The EVE dataset is a video dataset of Point-of-Gaze (on a 25 inch computer display) during the viewing of image, video, and text-based (wikipedia) stimuli.
It consists of 54 participants (39 train / 5 validation / 10 test), with video taken from 4 different views while they gaze upon the presented stimuli.

For further information, and to gain access to the dataset, please follow the instructions at the [EVE project page](https://ait.ethz.ch/projects/2020/EVE/).

## Dataset folder structure

On receipt (and download) of the dataset, you will find that it consists of 54 folders, each corresponding to a single dataset participant.
The following folders should exist:

```
train01, train02, ..., train39

val01, val02, ..., val05

test01, test02, ..., test10
```

Each folder will consist of subfolders with names similar to the following:
```
step008_image_MIT-i2263021117
step009_image_MIT-i2267703789
step010_image_MIT-istatic-outdoor-street-city-cambridge-uk-IMG-8893
step011_image_MIT-i1325514089
...
step030_video_diem-harry-potter-6-trailer
step031_video_diem-movie-trailer-ice-age-3
step032_video_vagba-track01
...
step116_video_Wikimedia-Washlets-high-tech-toilets-in-Japan
step117_video_Wikimedia-Barack-Obama-inaugural-address
step120_wikipedia_wikipedia-random
```
which corresponds to approximately:
* 60 image stimuli (each exposed for 3 seconds)
* 12-minutes worth of video stimuli
* 3x 2-minute Wikipedia gazing

The image and video stimuli are sourced from the following
* [image] MIT - https://people.csail.mit.edu/tjudd/WherePeopleLook/index.html
* [video] DIEM - https://thediemproject.wordpress.com/videos-and%C2%A0data/
* [video] VAGBA - https://stefan.winkler.site/resources.html
* [video] Kurzhals et al. - https://www.visus.uni-stuttgart.de/publikationen/benchmark-eyetracking
* 23 additional videos taken from [Wikimedia Commons](https://commons.wikimedia.org/wiki/Main_Page)
* and [random pages](https://en.m.wikipedia.org/wiki/Special:Random#/random) shown from the English Wikipedia

## Stimulus folder contents
In each *stimulus* folder, one can find data files pertaining to the 4 different camera views (`basler`, `webcam_l`, `webcam_c`, `webcam_r`):
* **`{camera}`.mp4**: the video recording after removal of camera distortion
* **`{camera}`_eyes.mp4**: a video of the 2 eyes, after "data normalization"
* **`{camera}`_face.mp4**: a video of the face, after "data normalization"
* **`{camera}`.timestamps.txt**: a file containing the timestamps of each frame present in the MP4 files above
* **`{camera}`.h5**: an HDF archive containing intermediate values from pre-processing and the ground-truth labels associated with the above files

In addition, we provide information regarding the screen content with the following files:
* **screen.mouse.txt**: a file where each line contains `<timestamp> <mouse x-position in pixels> <mouse y-position in pixels>`
* **screen.mp4**: a full 1080p recording of the screen
* **screen.128x72.mp4**: a downscaled version of the full video for the purpose of faster data loading during training
* **screen.timestamps.txt**: a file containing the timestamps of each frame present in the MP4 files above

### HDF file format

The following keys are provided directly as shown (presented with `name` - `shape of array`):

* `camera_matrix` - `(3, 3)` - camera calibration matrix using the pinhole camera model
* `camera_transformation` - `(4, 4)` - the full transformation to bring a point from the screen coordinate system to the given camera's coordinate system
* `inv_camera_transformation` - `(4, 4)` - the inverse of the previous line
* `millimeters_per_pixel` - `(2, )` - the x/y scaling factors to convert pixels to millimeters
* `pixels_per_millimeter`- `(2, )` - the inverse of the previous line

The following values consist of both the actual values (under the field `data`) and whether the value is valid as described by the Tobii firmware, and via failures in the data pre-processing procedure (under the field `validity`).
The shapes described below are of the actual values (under the field `data`).

* `facial_landmarks` - `(N, 68, 2)` - facial landmarks (3D prediction, but only u,v retained) detected using [FAN](https://github.com/1adrianb/face-alignment) wrt the full camera frame
* `head_rvec` - `(N, 180, 3, 1)` - the rotation of the head as determined via [`cv2.solvePnP`](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d)
* `head_tvec` - `(N, 180, 3, 1)` - the translation of the head as determined via [`cv2.solvePnP`](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d)
* `left_p` - `(N, )` - pupil size in millimeters of the left eye
* `right_p` - `(N, )` - pupil size in millimeters of the right eye

The so-called **data normalization** procedure (described further later) is used to yield image patches specifically of the face, left-eye and right-eye.
This procedure yields the following values, where `N` is the number of frames:

* `{face,left,right}_PoG_tobii` - `(N, 2)` - on-screen pixel coordinates for Point-of-Gaze as estimated by the Tobii Pro Spectrum device
* `{face,left,right}_g_tobii` - `(N, 2)` - the roll-removed gaze direction after data normalization in spherical coordinates
* `{face,left,right}_R` - `(N, 3, 3)` - the rotation correction applied to the raw gaze direction vector (line 72 of example code)
* `{face,left,right}_W` - `(N, 3, 3)` - the perspective transform matrix (line 74 of example code)
* `{face,left,right}_h` - `(N, 2)` - the roll-removed head orientation after data normalization in spherical coordinates
* `{face,left,right}_o` - `(N, 3)` - the 3D origin of gaze as defined for the particular patch (face, left-eye, or right-eye)

## A note on the **data normalization** procedure

For a detailed explanation of this procedure, please refer to the paper, example code, and materials provided at https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/revisiting-data-normalization-for-appearance-based-gaze-estimation/

Simply speaking, this procedure provides improved consistency when producing eye or face patches via careful control of a **virtual** camera. In [this paper](https://dl.acm.org/doi/10.1145/3204493.3204548) by **Zhang et al.** an improvement was proposed to the data normalization method, and thus we use this adjusted method in processing EVE's ground-truth.

Please refer to the official example Python script for data normalization (https://www.mpi-inf.mpg.de/fileadmin/inf/d2/xucong/data_normalization_code.zip) to find the corresponding measures as shown above, demonstrated by code.

## A note on the timestamps
For the purpose of this dataset, all timestamps may be considered as being synchronized.
However, our camera synchronization is a *best effort* and due to the hardware and firmware involved, reliable synchronization exists only between the `basler` camera and the eye tracking data.
The webcams (`webcam_l`, `webcam_c`, `webcam_r`) were synchronized via the system timestamp as provided by the Video4Linux driver and may be imprecise.
