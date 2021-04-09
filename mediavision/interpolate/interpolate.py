#!/usr/bin/env python3
# -*- coding = utf-8 -*-
from mediavision.core.types import ImageOrPath
from mediavision.core.model_registry import get_model
from mediavision.core.processing_registry import get_processor

def interpolate(path: ImageOrPath, output_path: ImageOrPath = None,
                model: str = "rife_net", **kwargs) -> None:
   """Recolor a provided image."""
   # Load in the model.
   model = get_model(model)(pretrained = True)

   # Choose either the image or video processor.
   processor = get_processor('rife_interpolator_video')

   # Interpolate the video.
   processor(path, output_path, model, **kwargs)

