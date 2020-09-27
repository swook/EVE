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

import numpy as np

from core import DefaultConfig

config = DefaultConfig()
logger = logging.getLogger(__name__)


def gaussian_2d(shape, centre, sigma=1.0):
    """Generate heatmap with single 2D gaussian."""
    xs = np.arange(0.5, shape[1] + 0.5, step=1.0, dtype=np.float32)
    ys = np.expand_dims(np.arange(0.5, shape[0] + 0.5, step=1.0, dtype=np.float32), -1)
    alpha = -0.5 / (sigma**2)
    heatmap = np.exp(alpha * ((xs - centre[0])**2 + (ys - centre[1])**2))
    return heatmap


def onehot_from_values(v, v_min, v_max, n_bins, clipped=False):
    if clipped:
        v = np.clip(v, a_min=v_min+1e-6, a_max=v_max-1e-6)
    v = ((v - (v_min+1e-6)) / (v_max - v_min))  # values in [0, 1]
    hmap = gaussian_2d([n_bins, n_bins], v * n_bins, sigma=config.onehot_sigma)
    hmap = hmap.flatten()
    hmap *= 1.0 / np.sum(hmap)
    return hmap
