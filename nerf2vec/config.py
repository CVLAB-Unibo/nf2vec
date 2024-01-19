"""
# ####################
# NERF2VEC
# ####################
"""
#
# DIMENSIONS
#
ENCODER_EMBEDDING_DIM = 1024
ENCODER_HIDDEN_DIM = [512, 512, 1024, 1024]


DECODER_INPUT_DIM = 3
DECODER_HIDDEN_DIM = 1024
DECODER_NUM_HIDDEN_LAYERS_BEFORE_SKIP = 2
DECODER_NUM_HIDDEN_LAYERS_AFTER_SKIP = 2
DECODER_OUT_DIM = 4

# 
# TRAIN 
#
NUM_EPOCHS = 501
BATCH_SIZE = 16
LR = 1e-4
WD = 1e-2
BG_WEIGHT = 0.2
FG_WEIGHT = 1 - BG_WEIGHT

"""
# ####################
# NERFACC
# ####################
"""
#
# GRID
#
import os
try:
    from nerfacc import ContractionType
    GRID_CONTRACTION_TYPE = ContractionType.AABB
except ImportError:
    pass
GRID_AABB = [-0.7, -0.7, -0.7, 0.7, 0.7, 0.7]
GRID_RESOLUTION = 96
GRID_CONFIG_N_SAMPLES = 1024

GRID_RECONSTRUCTION_TOTAL_ITERATIONS = 20
GRID_RECONSTRUCTION_WARMUP_ITERATIONS = 5
GRID_NUMBER_OF_CELLS = 884736  # (884736 if resolution == 96,  2097152 if resolution == 128)
GRID_BACKGROUND_CELLS_TO_SAMPLE = 32000

#
# RAYS
#
NUM_RAYS = 55000
MAX_FOREGROUND_COORDINATES = 25000
MAX_BACKGROUND_COORDINATES = 10000

#
# INSTANT-NGP 
#
MLP_INPUT_SIZE = 3
MLP_ENCODING_SIZE = 24
MLP_INPUT_SIZE_AFTER_ENCODING = MLP_INPUT_SIZE * MLP_ENCODING_SIZE * 2
MLP_OUTPUT_SIZE = 4
MLP_HIDDEN_LAYERS = 3
MLP_UNITS = 64

INSTANT_NGP_MLP_CONF = {
    'aabb': GRID_AABB,
    'unbounded':False,
    'encoding':'Frequency',
    'mlp':'FullyFusedMLP',
    'activation':'ReLU',
    'n_hidden_layers':MLP_HIDDEN_LAYERS,
    'n_neurons':MLP_UNITS,
    'encoding_size':MLP_ENCODING_SIZE
}

INSTANT_NGP_ENCODING_CONF = {
    "otype": "Frequency",
    "n_frequencies": 24
}

NERF_WEIGHTS_FILE_NAME = 'nerf_weights.pth'

#
# TINY-CUDA
#
TINY_CUDA_MIN_SIZE = 16

"""
# ####################
# LOGGING
# ####################
"""
WANDB_CONFIG = {
    'ENCODER_EMBEDDING_DIM': ENCODER_EMBEDDING_DIM,
    'ENCODER_HIDDEN_DIM': ENCODER_HIDDEN_DIM,
    'DECODER_INPUT_DIM': DECODER_INPUT_DIM,
    'DECODER_HIDDEN_DIM': DECODER_HIDDEN_DIM,
    'DECODER_NUM_HIDDEN_LAYERS_BEFORE_SKIP': DECODER_NUM_HIDDEN_LAYERS_BEFORE_SKIP,
    'DECODER_NUM_HIDDEN_LAYERS_AFTER_SKIP': DECODER_NUM_HIDDEN_LAYERS_AFTER_SKIP,
    'DECODER_OUT_DIM': DECODER_OUT_DIM,
    'NUM_EPOCHS': NUM_EPOCHS,
    'BATCH_SIZE': BATCH_SIZE,
    'LR': LR,
    'WD': WD,
    "NUM_RAYS": NUM_RAYS,
    "GRID_RESOLUTION": GRID_RESOLUTION
}


"""
# ####################
# DATASET
# ####################
"""
TRAIN_SPLIT = 'train'
VAL_SPLIT = 'val'
TEST_SPLIT = 'test'


LABELS_TO_IDS = {
    "02691156": 0,   # airplane
    "02828884": 1,   # bench
    "02933112": 2,   # cabinet
    "02958343": 3,   # car
    "03001627": 4,   # chair
    "03211117": 5,   # display
    "03636649": 6,   # lamp
    "03691459": 7,   # speaker
    "04090263": 8,   # rifle
    "04256520": 9,   # sofa
    "04379243": 10,  # table
    "04401088": 11,  # phone
    "04530566": 12   # watercraft
}

# TODO: COMMENT THESE!
#'02992529': 4, tablet delete?
#"03948459": 9, gun delete?

NUM_CLASSES = len(LABELS_TO_IDS)
