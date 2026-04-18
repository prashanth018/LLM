from utils.constants import GPT2_MODEL_DIRECTORY, GPT2_MODEL_SIZE
from utils.gpt_download import download_and_load_gpt2


def load_gpt2():
    download_and_load_gpt2(GPT2_MODEL_SIZE, GPT2_MODEL_DIRECTORY)


if __name__ == "__main__":
    load_gpt2()
