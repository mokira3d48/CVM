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
import sys
import logging
import traceback
import argparse

import torch
import torch.nn.functional as F
from torch import nn

from torch.utils import data
from torch.utils.data import Dataset as BaseDataset


###############################################################################
# DATASET
###############################################################################

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    format='%(asctime)s - - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vae_train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_argument():
    """
    Function to return command line argument parsed

    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-dt', '--data-train',
                        type=str, help="Data train path", required=True)
    parser.add_argument('-dv', '--data-val',
                        type=str, help="Data validation path", required=True)
    args = parser.parse_args()

    logger.info("Arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    return args


def main():
    """
    Main function to run training process
    """
    args = get_argument()


if __name__ == '__main__':

    def print_err():
        """
        Function of Error tracked printing
        """
        # get traceback error
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tbobj = traceback.extract_tb(exc_traceback)

        # msg += ''.join([' ' for _ in ERRO]) +
        #     "\t%16s %8s %64s\n" % ("FILE NAME", "LINENO", "MODULE/FUNCTION",);
        for tb in tbobj:
            logger.error(
                "\t%16s %8d %64s\n" % (tb.name, tb.lineno, tb.filename,))

    try:
        main()
        exit(0)
    except KeyboardInterrupt as e:
        print("\033[91mCanceled by user!")
        exit(125)
    except FileNotFoundError:
        print_err()
        exit(2)
    except Exception:  # noqa
        print_err()
        exit(1)
