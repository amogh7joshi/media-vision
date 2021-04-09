#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import shutil
import subprocess
from typing import Any

from mediavision.system import is_ffmpeg_installed

def transfer_audio(source_video: Any, target_video: Any) -> None:
   """Transfer the audio of a source video to the target video."""
   # First, validate that FFMPEG is installed, otherwise this method
   # will not be able to perform any actions and will instead result
   # in a cryptic error message (that is not really useful).
   if not is_ffmpeg_installed():
      print("\033[91m" + "[FRAME INTERPOLATION]: " +
            "Cannot perform audio transfer - FFMPEG is not installed." + "\033[0m")

   # Create the temporary audio file.
   if not os.path.exists('./temp/'):
      os.makedirs('./temp')
   temp_output = os.path.join('./temp', 'audio.mkv')

   # Remove the temporary directory if it already exists.
   if os.path.isdir("temp"):
      shutil.rmtree("temp")
   # Create the temporary directory again.
   os.makedirs("temp")
   
   # Copy the audio from the source video to the temporary file.
   os.system('ffmpeg -y -i "{}" -c:a copy -vn {} >/dev/null 2>&1'
                           .format(source_video, temp_output))

   # Create the no-audio output video.
   target_no_audio = os.path.splitext(target_video)[0] + "_noaudio" + \
                     os.path.splitext(target_video)[1]
   os.rename(target_video, target_no_audio)

   # Try to combine the audio and video files.
   os.system('ffmpeg -y -i "{}" -i {} -c copy "{}" >/dev/null 2>&1'
                           .format(target_no_audio, temp_output, target_video))

   # Check whether FFMPEG failed to merge the audio and video and
   # try converting the audio to AAC instead, see if that works.
   if os.path.getsize(target_video) == 0:
      # Create a new temporary file.
      temp_output = os.path.join('./temp', 'audio.m4a')
      # Re-try moving the audio over.
      os.system('ffmpeg -y -i "{}" -c:a aac -b:a 160k -vn {} >/dev/null 2>&1'
                              .format(source_video, temp_output))
      os.system('ffmpeg -y -i "{}" -i {} -c copy "{}" >/dev/null 2>&1'
                              .format(target_no_audio, temp_output, target_video))
      if os.path.getsize(target_video) == 0:
         # If that fails too, then exit without transferring the video.
         os.rename(target_no_audio, target_video)
         print("\033[91m" + "[FRAME INTERPOLATION]: " +
               "Audio transfer failed. Output video will have no audio." + "\033[0m")
      else:  # Still warn the user that lossless audio transfer failed.
         print("\033[91m" + "[FRAME INTERPOLATION]: " +
               "Lossless audio transfer failed. Audio was transcoded to AAC (M4A)." + "\033[0m")
         # Remove the video without audio.
         os.remove(target_no_audio)
   else:
      # Remove the video without audio.
      os.remove(target_no_audio)
      # Print a success message.
      print("\033[92m" + "[FRAME INTERPOLATION]: " + "Audio transfer successful." + "\033[0m")

   # Remove the temporary directory.
   shutil.rmtree("temp")

