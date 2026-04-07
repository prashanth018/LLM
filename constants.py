BATCH_SIZE = 8
VECTOR_DIM = 18
NUM_HEADS = 6
NUM_LAYERS = 12
CONTEXT_LENGTH_SMALL = 4
CONTEXT_LENGTH_MEDIUM = 40
STRIDE_SMALL = CONTEXT_LENGTH_SMALL
STRIDE_MEDIUM = CONTEXT_LENGTH_MEDIUM
MHA_DIM_IN = VECTOR_DIM
MHA_DIM_OUT = VECTOR_DIM
DROPOUT = 0.1

CONFIG_EXP_S = {
    "context_length": CONTEXT_LENGTH_SMALL,
    "stride": STRIDE_SMALL,
    "batch_size": BATCH_SIZE,
    "dim": VECTOR_DIM,
    "n_heads": NUM_HEADS,
    "n_layers": NUM_LAYERS,
    "dropout": DROPOUT,
}

CONFIG_EXP_M = {
    "context_length": CONTEXT_LENGTH_MEDIUM,
    "stride": STRIDE_MEDIUM,
    "batch_size": BATCH_SIZE,
    "dim": VECTOR_DIM,
    "n_heads": NUM_HEADS,
    "n_layers": NUM_LAYERS,
    "dropout": DROPOUT,
}
