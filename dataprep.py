import tiktoken
from torch.utils.data import DataLoader


from dataset import GutenbergDataset
from nnets import PositionalEmbedding, TokenEmbedding
from torch import tensor

CONTEXT_LENGTH = 4
STRIDE = 4
VECTOR_DIM = 3
BATCH_SIZE = 8


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GutenbergDataset(
        tokenizer=tokenizer, context_length=CONTEXT_LENGTH, stride=STRIDE
    )
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    inputs, targets = next(iter(dataloader))
    # print("Inputs = ", inputs)
    # print("Target = ", targets)
    print(inputs.shape)
    vocab_size = tokenizer.n_vocab
    token_embedding = TokenEmbedding(vocab_size, VECTOR_DIM)
    positional_embedding = PositionalEmbedding(CONTEXT_LENGTH, VECTOR_DIM)

    # temp = tensor([[1], [2]])
    # temp = tensor([[1, 2]])
    # temp = tensor([1, 2])
    # temp = tensor([[1, 2, 3, 4], [4, 5, 6, 4]])
    # print(temp.shape)
    # print(token_embedding(temp).shape)

    input_batch_token_embedding = token_embedding(inputs)
    # print(input_batch_token_embedding)
    print(
        "Shape of Token Embedding for the current batch = ",
        input_batch_token_embedding.shape,
    )
    input_batch_positional_embedding = positional_embedding()
    print("Shape of Positional Embedding = ", input_batch_positional_embedding.shape)
    input_embeddings = input_batch_token_embedding + input_batch_positional_embedding
    print("Shape of Input Embedding for the current batch = ", input_embeddings.shape)
