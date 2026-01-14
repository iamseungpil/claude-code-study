"""Configuration - HARDCODED VALUES
TODO: Move to config.json
"""

# Training configuration
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
WEIGHT_DECAY = 0.0

# Model configuration
HIDDEN_SIZE = 128
DROPOUT_RATE = 0.25
NUM_CLASSES = 10

# Data configuration
DATA_DIR = 'data/raw'
OUTPUT_DIR = 'outputs'

# Device
DEVICE = 'auto'  # 'cpu', 'cuda', or 'auto'

# Logging
LOG_INTERVAL = 100
VERBOSE = True

# Seed
RANDOM_SEED = 42

# Paths - some of these don't exist
MODEL_PATH = 'outputs/model_final.pt'
LOG_PATH = 'logs/train.log'
CHECKPOINT_DIR = 'outputs/checkpoints'

# Old settings - kept for backwards compatibility
OLD_BATCH_SIZE = 32
OLD_LR = 0.01
DEPRECATED_FLAG = True
