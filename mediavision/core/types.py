#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
from typing import Union

import numpy as np

import torch
import torch.nn as nn

from mediavision.core.base import MediaVisionModelBase

# A type to represent either images, image paths, or
# strings which themselves are image paths.
ImageOrPath = Union[np.ndarray, str, os.PathLike]

# A type to represent all kinds of paths, whether actual
# PathLike objects, strings, or byte objects.
AnyPath = Union[str, bytes, os.PathLike]

# A type representing any kind of model in the toolbox.
AnyModel = MediaVisionModelBase

# A return type representing a float or a torch Tensor.
TensorReturnType = Union[torch.Tensor, float]
