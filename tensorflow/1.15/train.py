import os
import sys
import time
import argparse
import tensorflow as tf
import numpy as np
import imageio
import random
from utils import utils
from network import Network

def main(args):
    BEST = 0.0

    config_file = args.config_file
    # I/O
    config = utils.import_file(config_file, "config")

    network = Network()
    network.initialize(config, config.num_classes)

    #
    # Main Loop
    #
    print(
        "\nStart Training\n# epochs: %d\nepoch_size: %d\nbatch_size: %d\n"
        % (config.num_epochs, config.epoch_size, config.batch_size)
    )
    global_step = 0
    start_time = time.time()
    for epoch in range(config.num_epochs):

        # Training
        for step in range(config.epoch_size):
            
            train_imgs = np.ndarray((config.batch_size, 240, 240, 1), np.float32)
            train_labels =  np.random.randint(config.num_classes, size=(config.batch_size))

            
            wl, sm, global_step = network.train(
                    train_imgs,
                    train_labels,
                    config.lr,
                    config.keep_prob
                )
            
            wl["lr"] = config.lr

            # Display
            if step % config.summary_interval == 0:
                duration = time.time() - start_time
                start_time = time.time()
                utils.display_info(epoch, step, duration, wl)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", help="The path to the training configuration file", type=str
    )
    args = parser.parse_args()
    main(args)
