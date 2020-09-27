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
from tensorboardX import SummaryWriter
import torchvision

import logging
logger = logging.getLogger(__name__)


class Tensorboard(object):

    __instance = None
    __writer = None
    __current_step = 0
    __output_dir = None

    # Make this a singleton class
    def __new__(cls, *args):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self, output_dir, model=None, input_to_model=None):
        self.__output_dir = output_dir
        self.__writer = SummaryWriter(output_dir)
        if model is not None:
            self.add_graph(model, input_to_model)

    def __del__(self):
        self.__writer.close()

    @property
    def output_dir(self):
        return self.__output_dir

    def update_current_step(self, step):
        self.__current_step = step

    def add_graph(self, model, input_to_model=None):
        self.__writer.add_graph(model, input_to_model)

    def add_grid(self, tag, values):
        grid = torchvision.utils.make_grid(values)
        self.__writer.add_image(tag, grid, self.__current_step)

    def add_scalar(self, tag, value):
        self.__writer.add_scalar(tag, value, self.__current_step)

    def add_image(self, tag, value):
        self.__writer.add_image(tag, value, self.__current_step)
