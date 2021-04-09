#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import sys
import json
import subprocess

# Create the colorful printing log variables.
RED_C = "\033[91m"
END_C = "\033[0m"

# Construct and validate the current directory.
cwd = os.getcwd()
if not cwd.endswith('/scripts'):
   sys.stdout.write(RED_C + f"You should be in the `scripts` directory, instead "
                            f"you are in the {cwd} directory. Try re-downloading "
                            f"and re-configuring the repository." + END_C)
   sys.exit(1)

# Try entering the path to the models directory.
try:
   os.chdir(os.path.join(os.path.dirname(cwd), 'mediavision/models'))
except OSError as ose:
   sys.stdout.write(RED_C + str(ose) + END_C)
   sys.exit(1)

# Check whether there are models.
if len([item for item in os.listdir(os.getcwd()) if item != '.DS_Store']) != 2:
   sys.stdout.write(RED_C + f"There should be either two model files or a model file "
                            f"and a loaded model directory, instead there are not "
                            f"two objects in the current repository. Re-download "
                            f"the model files and try again" + END_C)
   sys.exit(1)

# Unzip the zipped RIFE file if it exists.
rife_model_file = os.path.join(os.getcwd(), 'RIFE_trained_model_HDv2.zip')
if 'RIFE_trained_model_HDv2.zip' in [item for item in os.listdir(os.getcwd()) if item != '.DS_Store']:
   status = subprocess.check_call(['unzip', rife_model_file],
                                  stdout = subprocess.PIPE)

   # Ensure that the file was properly unzipped or else raise an error.
   if status != 0:
      sys.stdout.write(RED_C + f"There was an error in trying to unzip the RIFE model. "
                               f"Please redownload it and try again." + END_C)

# Create any additional model files.
esrgan_model_file = os.path.join(os.getcwd(), 'RRDB_ESRGAN_x4.pth')

# Move the different weights files to the JSON file.
weights_json = os.path.join(os.path.dirname(os.getcwd()), 'weights', 'weights.json')
with open(weights_json, 'r') as wj:
   weights_dict = json.load(wj)
weights_dict['rife_nets'] = rife_model_file
weights_dict['esrgan_rrdbnet'] = esrgan_model_file
with open(weights_json, 'w') as wj:
   dump = json.dumps(weights_dict)
   wj.write(dump)




