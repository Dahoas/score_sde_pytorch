# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training and evaluation"""

import run_lib
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os
import tensorflow as tf

import argparse


if __name__ == "__main__":
  #app.run(main)
  parser = argparse.ArgumentParser()
  #parser.add_argument("--config_path", type=str, required=True)
  #parser.add_argument("--workdir", type=str, required=True)
  parser.add_argument("--mode", type=str, required=True)
  args = parser.parse_args()
  #parser.add_argument("--eval_folder", type=str, default="eval")

  from configs.vp.ddpm import cifar10_continuous as config_py
  #from configs.vp import cifar10_ncsnpp_continuous_fft as config_py
  #from configs.vp import cifar10_ncsnpp_continuous as config_py
  config = config_py.get_config()

  #workdir = "checkpoints/vp/cifar10_ddpm_continuous"
  workdir = "train_temp"
  if args.mode == "train":
    # Create the working directory
    tf.io.gfile.makedirs(workdir)
    # Set logger so that it outputs to both console and file
    # Make logging work for both disk and Google Cloud Storage
    gfile_stream = open(os.path.join(workdir, 'stdout.txt'), 'w')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')
    # Run the training pipeline
    run_lib.train(config, workdir)
  elif args.mode == "eval":
    run_lib.evaluate(config, workdir, "eval")
  else:
    raise ValueError("Unsupported mode: {}".format(args.mode))
