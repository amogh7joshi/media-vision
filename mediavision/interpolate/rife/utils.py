#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import time
import _thread
import warnings
import subprocess
from queue import Queue
from functools import partial
from typing import Tuple, Any, List

import cv2
import numpy as np

import skvideo.io
from tqdm import tqdm

import torch
import torch.nn.functional as F

from mediavision.core.types import AnyPath
from mediavision.core.base import MediaVisionModelBase
from mediavision.core.console import Console
from mediavision.core.processing_registry import register_processor
from mediavision.interpolate.rife.ssim import ssim_matlab
from mediavision.interpolate.rife.audio import transfer_audio
from mediavision.system import has_audio

def get_video_properties(video_path: AnyPath) -> Tuple[int, int, Any, Any]:
   """Loads a video and constructs an output video writer."""
   # Construct the video.
   vid = cv2.VideoCapture(video_path)

   # Get the parameters of the video.
   fps = vid.get(cv2.CAP_PROP_FPS)
   n_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
   vid.release()

   # Get the last frame of the video.
   video_generator = skvideo.io.vreader(video_path)
   last_frame = next(video_generator)

   # Return the video properties.
   return fps, n_frames, last_frame, video_generator

def build_read_buffer(read_buffer: Queue, video_generator: Any, montage: bool = True,
                      left: int = None, w: int = None) -> None:
   """Builds a read buffer and constructs the image montage."""
   try:
      # Iterate over each frame of the video.
      for frame in video_generator:
         # Build the montage if requested to.
         if montage:
            frame = frame[:, left: left + w]
         # Add the frame to the read buffer.
         read_buffer.put(frame)
   except:
      # Continue if it is not possible to parse over the video.
      pass

   # Add a value to the end.
   read_buffer.put(None)

def make_inference(image0: torch.Tensor, image1: torch.Tensor,
                   exp: int, scale: float, model: MediaVisionModelBase) -> List:
   """Inference the result of the input images from the model."""
   # Get the middle inference.
   middle = model.inference(image0, image1, scale) # noqa

   # If the reduction value is one, then return the current frame.
   if exp == 1:
      return [middle]

   # Otherwise, keep recursing through until all new frames are gathered.
   first_half = make_inference(image0, middle, exp - 1, scale, model)
   second_half = make_inference(middle, image1, exp - 1, scale, model)

   # Return the complete list of frames.
   return [*first_half, middle, *second_half]

def pad_image(image: torch.Tensor, padding: tuple, fp16: bool = False) -> torch.Tensor:
   """Pad an image."""
   if fp16:
      return F.pad(image, padding).half()
   return F.pad(image, padding)

@register_processor(name = 'rife_interpolator_image')
def rife_interpolator_image_processor(images: np.ndarray,
                                      output_path: AnyPath,
                                      model: MediaVisionModelBase,
                                      exp: int = 4, ratio: int = 0,
                                      ratio_threshold: float = 0.02,
                                      max_cycles: int = 8, dst_fps: int = 30,
                                      construct_video: bool = True):
   """Process and interpolate an image through a rife model."""
   # Ignore the parameter warnings from PyTorch.
   warnings.filterwarnings("ignore")

   # Initialize the console for output logging.
   console = Console(header = "[FRAME INTERPOLATION]")

   # Configure the device.
   device = ('cuda' if torch.cuda.is_available() else 'cpu')

   # Extract the images from the paths.
   image_0, image_1 = images

   # Ensure the two images have the same shape.
   if image_0.shape != image_1.shape:
      raise ValueError(
         f"Image shapes are not the same, got {image_0.shape} and {image_1.shape}")

   # Convert the images to tensors.
   image_0 = torch.tensor(image_0.transpose(2, 0, 1) / 255.0) \
                  .to(device).unsqueeze(0)
   image_1 = torch.tensor(image_1.transpose(2, 0, 1) / 255.0) \
                  .to(device).unsqueeze(0)

   # Construct the padding arguments for the images.
   n, c, h, w = image_0.shape
   pad_h = ((h - 1) // 32 + 1) * 32
   pad_w = ((w - 1) // 32 + 1) * 32
   padding = (0, pad_w - w, 0, pad_h - h)

   # Pad the images.
   image_0 = F.pad(image_0, padding)
   image_1 = F.pad(image_1, padding)

   # Create a list of interpolated images.
   interpolated_images = [image_0]

   # Interpolate the images.
   if ratio != 0: # For the case in which an inference ratio is provided.
      # Construct the ratios for each image.
      image_0_ratio, image_1_ratio = 0.0, 1.0

      # Create the different cases for finding intermediate images.
      if ratio <= image_0_ratio + ratio_threshold / 2:
         # The `middle` image is simply the first image.
         middle = image_0_ratio
      elif ratio >= image_1_ratio - ratio_threshold / 2:
         # The `middle` image is simply the second image.
         middle = image_1_ratio
      else: # Otherwise, actually construct intermediate frames.
         # Create placeholders for the images.
         temp_image_0 = image_0
         temp_image_1 = image_1

         # Create a progress bar to track each cycle.
         p_bar = tqdm(total = max_cycles,
                      desc = console.header + "Constructing intermediate frames",
                      bar_format = "%s{l_bar}{bar}{r_bar}%s" % ('\033[94m', '\033[0m'))

         # Iterate over each intermediate frame to be made.
         for cycle in range(max_cycles):
            # Get the predicted frame.
            middle = model.inference(temp_image_0.float(), temp_image_1.float()) # noqa
            m_ratio = (image_0_ratio + image_1_ratio) / 2

            # Determine whether to continue or make a new frame.
            if ratio - (ratio_threshold / 2) <= m_ratio \
                  <= ratio + (ratio_threshold / 2):
               break
            if ratio > m_ratio:
               temp_image_0 = middle
               image_0_ratio = m_ratio
            else:
               temp_image_1 = middle
               image_1_ratio = m_ratio

            # Update the progress bar.
            p_bar.update(1)

         # Close the progress bar.
         p_bar.close()

      # Add the images to the list.
      interpolated_images.extend(middle, image_1) # noqa
   else: # Otherwise, add multiple sets of frames.
      # Create a progress bar to track each cycle.
      p_bar = tqdm(total = exp,
                   desc = console.header + "Constructing intermediate frames",
                   bar_format = "%s{l_bar}{bar}{r_bar}%s" % ('\033[94m', '\033[0m'))

      # Add the last image to the list.
      interpolated_images.append(image_1)

      # Iterate over the provided parameter.
      for i in range(exp):
         # Create a temporary list.
         temp_list = []

         # Iterate over the images in the list.
         for j in range(len(interpolated_images) - 1):
            # Inference the middle image.
            middle = model.inference( # noqa
               interpolated_images[j].float(), interpolated_images[j + 1].float())

            # Add the images to the list.
            temp_list.append(interpolated_images[j])
            temp_list.append(middle)

         # Add the final image to the list.
         temp_list.append(image_1)

         # Reset the interpolated image list.
         interpolated_images = temp_list

         # Update the progress bar.
         p_bar.update(1)

      # Close the progress bar.
      p_bar.close()

   # Create the output directory.
   if '.' not in output_path: # Create the output image directory.
      os.makedirs(output_path) if not os.path.exists(output_path) else None
   else: # Create the directory for the video.
      os.makedirs(os.path.dirname(output_path)) \
         if not os.path.exists(os.path.dirname(output_path)) else None

   # Get the base ID of the input.
   base = "interpolated"

   # Create a directory with saved images.
   for img in range(len(interpolated_images)):
      cv2.imwrite(os.path.join(output_path, f'{base}_{img}.png'),
                  (interpolated_images[img][0] * 255)
                  .byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])

   # If asked to create a video, construct the GIF.
   if construct_video:
      # Remove the existing video.
      os.remove(os.path.join(output_path, 'interpolated_video.gif')) \
         if os.path.exists(os.path.join(output_path, 'interpolated_video.gif')) else None

      # Execute the construction command.
      os.system(" ".join((
         'ffmpeg', '-r', '10', '-f', 'image2', '-i',
         f'{output_path}/interpolated_%d.png', '-s', f'{w}x{h}', '-vf',
         '"split[s0][s1];[s0]palettegen=stats_mode=single[p];[s1][p]paletteuse=new=1"',
         f'{output_path}/interpolated_video.gif', '-hide_banner', '-loglevel error')))

@register_processor(name = 'rife_interpolator_video')
def rife_interpolator_video_processor(video_path: AnyPath, output_path: AnyPath = None,
                                      model: MediaVisionModelBase = None,
                                      exp: int = None, dst_fps: int = None,
                                      scale: float = 0.5, skip: bool = True,
                                      out_ext: str = None, png: bool = False,
                                      out_png_path: str = None, UHD: bool = True,
                                      montage: bool = False):
   """Process and interpolate a video through a rife model."""
   # Ignore the parameter warnings from PyTorch.
   warnings.filterwarnings("ignore")

   # Ensure that the video path exists.
   if not os.path.exists(video_path):
      raise FileNotFoundError(f"The input video path '{video_path}' does not exist.")

   # Configure the device.
   device = ('cuda' if torch.cuda.is_available() else 'cpu')

   # Get the GPU parameters.
   if torch.cuda.is_available():
      fp16 = True
   else:
      fp16 = False

   # Process the input parameters.
   if UHD and scale == 1.0:
      # Allow for processing UHD video (4K, etc.).
      scale = 0.5
   # Ensure a valid scale has been provided.
   assert scale in [0.25, 0.5, 1.0, 2.0, 4.0], f"Invalid scale {scale} provided."
   # Check the PNG parameter and whether an output path has been provided.
   if png and out_png_path is None:
      raise NotADirectoryError("If you want to use the PNG parameter, you need to "
                               "provide a directory at which to write the images to.")
   elif png and not os.path.exists(out_png_path):
      try:
         os.makedirs(out_png_path)
      except:
         raise NotADirectoryError(f"The provided PNG output path directory "
                                  f"'{out_png_path}' does not exist.")

   # Get the video properties.
   video_fps, n_frames, last_frame, video_gen = get_video_properties(video_path)
   h, w, _ = last_frame.shape

   # Build the montage parameters.
   if montage:
      left = w // 4
      w = w // 2
      last_frame = last_frame[:, left: left + w]
   else:
      # For future processing.
      left = None

   # Calculate the amount to pad the image.
   tmp = max(32, int(32 / scale))
   ph = ((h - 1) // tmp + 1) * tmp
   pw = ((w - 1) // tmp + 1) * tmp
   padding = (0, pw - w, 0, ph - h)

   # Calculate the FPS of the output video.
   if dst_fps is None and exp is None:
      raise ValueError("You need to provide either a destination FPS or destination exp value.")
   if dst_fps is None:
      dst_fps = video_fps * (2 ** exp)
      # A tracker value for audio merging.
      fps_not_assigned = True
   else: # If the FPS value is already calculated, there is nothing to do.
      dst_fps = dst_fps
      fps_not_assigned = False

   # Get the video path and output extension.
   video_path_base, video_path_ext = os.path.splitext(video_path)
   if out_ext is None:
      out_ext = video_path_ext

   # Initialize the console for output logging.
   console = Console(header = "[FRAME INTERPOLATION]")

   # Print out information about the video.
   console.print(
      "{}{} - {} frames in total, interpolating from {:.3f} FPS to {:.3f} FPS.".format(
         os.path.basename(video_path_base), out_ext, n_frames, video_fps, dst_fps), color = "blue")

   # Print out information about audio merging.
   if not has_audio(video_path):
      # If the video has no audio to begin with, then there is nothing to merge.
      console.print("The video has no audio, no audio will be transferred.", color = "blue")
      has_no_audio = True
   else:
      has_no_audio = False
      if png is False and fps_not_assigned is True and not skip:
         console.print("The audio will be merged after the frame interpolation process.", color = "blue")
      else:
         console.print("Cannot merge audio, you are using the png, FPS, or skip flags.", color = "red")

   # Create the output video path.
   if output_path is not None:
      if not os.path.exists(os.path.dirname(output_path)):
         raise NotADirectoryError(f"The directory of the provided output "
                                  f"path '{output_path}' does not exist.")
      assert os.path.splitext(output_path)[1] == out_ext, \
         f"The output path has a different extension " \
         f"({os.path.splitext(output_path)[1]}) than the provided extension {out_ext}."
   else:
      # Create the output path.
      output_path = "{}_{}X_{}FPS{}".format(
         video_path_base, (2 ** exp), int(np.round(dst_fps)), out_ext)

   # Print information about the output path.
   console.print(f"Writing interpolated video to {output_path}.", color = "blue")

   # If the output path already exists, remove it first.
   if os.path.exists(output_path):
      console.print(f"Removing and overwriting existing file at {output_path}.", color = "blue")
      os.remove(output_path)

   # Construct the video writer.
   fourcc = cv2.VideoWriter_fourcc(*'mp4v')
   output_video = cv2.VideoWriter(output_path, fourcc, dst_fps, (w, h))

   # Create queues for processing the video.
   write_buffer = Queue(maxsize = 500)
   read_buffer = Queue(maxsize = 500)

   # The write buffer method will be defined inside of the main method in
   # order to have a constant video writer being used.
   def clear_write_buffer(write_buffer: Queue, png: bool = False, output_path: str = None) -> None:
      """Clear the output and write a single frame into the video."""
      count = 0
      while True:
         # Get the current frame.
         item = write_buffer.get()

         # Break if the video has reached its end.
         if item is None:
            break

         # Write the output if requested to.
         if png:
            cv2.imwrite(os.path.join(output_path, "{:0>7d}.png".format(count)),
                        item[:, :, ::-1])
            count += 1
         else:
            # Otherwise, write it to the video writer.
            output_video.write(item[:, :, ::-1])

   # Start the threads.
   _thread.start_new_thread(build_read_buffer, (read_buffer, video_gen, montage, left, w))
   _thread.start_new_thread(clear_write_buffer, (write_buffer, png, out_png_path))

   # Create a partial method for the padding method.
   static_pad = partial(pad_image, padding = padding, fp16 = fp16)

   # Create a partial method for the inference method.
   static_inference = partial(make_inference, scale = scale, model = model)

   # Process the last frame.
   image1 = torch.from_numpy(np.transpose(last_frame, (2, 0, 1))) \
               .to(device, non_blocking = True).unsqueeze(0).float() / 255.
   image1 = static_pad(image1)

   # Set the skipped frame counter to 1.
   skip_frame = 1

   # Create the progressbar.
   pbar = tqdm(total = int(n_frames),
               desc = console.header + "Interpolating Video",
               bar_format = "%s{l_bar}{bar}{r_bar}%s" % ('\033[94m', '\033[0m'))

   # Start iterating in reverse over the video.
   while True:
      # Get the next frame.
      frame = read_buffer.get()

      # Exit if there is no content (e.g., last frame).
      if frame is None:
         pbar.update(1)
         break

      # Process the current input frames.
      image0 = image1
      image1 = torch.from_numpy(np.transpose(frame, (2, 0, 1))) \
                    .to(device, non_blocking = True).unsqueeze(0).float() / 255.
      image1 = static_pad(image1)
      image0_small = F.interpolate(image0, (32, 32), mode = 'bilinear', align_corners = False)
      image1_small = F.interpolate(image1, (32, 32), mode = 'bilinear', align_corners = False)

      # Calculate the SSIM map.
      ssim = ssim_matlab(image0_small, image1_small)

      # Process the SSIM map and images.
      # There is a special case if the user wants to skip static frames, so determine
      # if the similarity between the frames makes them essentially static.
      if ssim > 0.995 and skip: # noqa
         # Warn the user if the skipped frames are altering the video.
         if skip_frame % 100 == 0:
            console.print(
               "\nWarning: Your video has {} static frames, skipping them"
               "may alter the duration of the original video.".format(skip_frame), color = "red")
         skip_frame += 1
         pbar.update(1)
         continue

      # Evaluate the SSIM map and construct a set of frames.
      if ssim < 0.5: # noqa
         # Create the parameters.
         output = []
         step = 1 / (2 ** exp)
         alpha = 0
         # Create the list of frames.
         for i in range((2 ** exp) - 1):
            alpha += step
            beta = 1 - alpha
            output.append(torch.from_numpy(
               np.transpose((cv2.addWeighted(
                  frame[:, :, ::-1], alpha, last_frame[:, :, ::-1], beta, 0)[:, :, ::-1].copy()),
                            (2, 0, 1))).to(device, non_blocking = True).unsqueeze(0).float() / 255.)
      else:
         # Otherwise, just get the regular model inference.
         output = static_inference(image0, image1, exp)

      # Create the montage as necessary.
      if montage:
         write_buffer.put(np.concatenate((last_frame, last_frame), 1))
         for mid in output:
            mid = (mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)
            write_buffer.put(np.concatenate((last_frame, mid[:h, :w]), 1))
      else:
         # Otherwise, just create the regular buffer.
         write_buffer.put(last_frame)
         for mid in output:
            mid = (mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)
            write_buffer.put(mid[:h, :w])

      # Update the progress bar and last frame.
      pbar.update(1)
      last_frame = frame

   # Create the final montage or general write output.
   if montage:
      write_buffer.put(np.concatenate((last_frame, last_frame), 1))
   else:
      write_buffer.put(last_frame)

   # Close the video and progress bar.
   while not write_buffer.empty():
      time.sleep(0.1)
   pbar.close()
   if output_video is not None:
      output_video.release()

   # Move the audio if possible.
   if not has_no_audio:
      if not png and fps_not_assigned is True and not skip:
         try:
            transfer_audio(video_path, output_path)
         except:
            print("\033[91m" + "[FRAME INTERPOLATION]: " +
                  "Audio transfer failed. Output video will have no audio." + "\033[0m")

   # Print the final set of information.
   console.print("Video interpolation complete.", color = "green")

   # Re-enable warnings.
   warnings.resetwarnings()




