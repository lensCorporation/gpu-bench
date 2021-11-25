''' Config Proto '''

import sys
import os


####### INPUT OUTPUT #######

# The interval between writing summary
summary_interval = 100

# Target image size for the input of network
image_size = [240, 240]

# 3 channels means RGB, 1 channel for grayscale
channels = 1

network = 'architecture.py'

embedding_size = 192

####### TRAINING STRATEGY #######

# Optimizer
optimizer = ("RMSPROP", {'momentum':0.9, 'decay':0.9, 'epsilon':1.0})

# Number of samples per batch
batch_size = 64

num_classes = 64

num_gpus = 1

# Number of batches per epoch
epoch_size = 1000

# Number of epochs
num_epochs = 500

# learning rate strategy
learning_rate_strategy = 'step'

# learning rate schedule
lr = 0.1
learning_rate_schedule = {
    0: 1 * lr,
}

# Multiply the learning rate for variables that contain certain keywords
learning_rate_multipliers = {
}

# Restore model
restore_model = None

# Keywords to filter restore variables, set None for all
restore_scopes = None

# Weight decay for model variables
weight_decay = 4e-5

# Keep probability for dropouts
keep_prob = 0.8


####### LOSS FUNCTION #######

# Scale for the logits
losses = {
    'softmax': {'weight_decay':1.0},
}

