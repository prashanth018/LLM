BATCH_SIZE = 8
VECTOR_DIM = 18
NUM_HEADS = 6
NUM_LAYERS = 12
CONTEXT_LENGTH_SMALL = 4
CONTEXT_LENGTH_MEDIUM = 40
BATCH_SIZE_M = 16
STRIDE_SMALL = CONTEXT_LENGTH_SMALL
STRIDE_MEDIUM = CONTEXT_LENGTH_MEDIUM
MHA_DIM_IN = VECTOR_DIM
MHA_DIM_OUT = VECTOR_DIM
DROPOUT = 0.1
NUM_EPOCHS = 5
GPT2_MODEL_SIZE = "124M"
GPT2_MODEL_DIRECTORY = "model_weights"
QWEN_INSTRUCT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

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
    "batch_size": BATCH_SIZE_M,
    "dim": VECTOR_DIM,
    "n_heads": NUM_HEADS,
    "n_layers": NUM_LAYERS,
    "dropout": DROPOUT,
}

CONFIG_GPT2_124M = {
    "context_length": 1024,
    "stride": 1024,
    "batch_size": 8,
    "dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "dropout": DROPOUT,
}
