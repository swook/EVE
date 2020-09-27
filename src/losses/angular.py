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
import numpy as np
import torch
import torch.nn.functional as F

from .base_loss_with_validity import BaseLossWithValidity
from models.common import pitchyaw_to_vector


class AngularLoss(BaseLossWithValidity):

    _to_degrees = 180. / np.pi

    def calculate_loss(self, a, b):
        a = pitchyaw_to_vector(a)
        b = pitchyaw_to_vector(b)
        sim = F.cosine_similarity(a, b, dim=1, eps=1e-8)
        sim = F.hardtanh_(sim, min_val=-1+1e-8, max_val=1-1e-8)
        return torch.acos(sim) * self._to_degrees
