#!/bin/env python3
# -*- encoding: utf-8 -*-

"""
===============================================================================
|                        YOLO Training IMPLEMENTATION                         |
===============================================================================


MIT License

Copyright (c) 2025 Dr Mokira

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

__version__ = '0.1.0'
__author__ = 'Dr Mokira'

import os
import logging

import torch
import torch.nn.functional as F
from torch import nn

from torch.utils import data
from torch.utils.data import Dataset as BaseDataset


def main():
    """
    Main function to run training process
    """
    ...


if __name__ == '__main__':
    try:
        main()
        exit(0)
    except KeyboardInterrupt as e:
        print("\033[91mCanceled by user!")
        exit(125)
