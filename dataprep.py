import tiktoken
from torch.utils.data import DataLoader


from dataset import GutenbergDataset
from nnets import PositionalEmbedding, TokenEmbedding
from attention import MultiHeadAttention, MultiHeadAttentionEfficient
from torch import tensor

CONTEXT_LENGTH = 4
STRIDE = 4
BATCH_SIZE = 8
NUM_HEADS = 6
VECTOR_DIM = 3
MHA_DIM_IN = VECTOR_DIM
MHA_DIM_OUT = 12
DROPOUT = 0.1


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GutenbergDataset(
        tokenizer=tokenizer, context_length=CONTEXT_LENGTH, stride=STRIDE
    )
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    inputs, targets = next(iter(dataloader))
    # print("Inputs = ", inputs)
    # print("Target = ", targets)
    print(inputs.shape)  # should be (8,4)
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
    )  # should be (8,4,3)
    input_batch_positional_embedding = positional_embedding()
    print(
        "Shape of Positional Embedding = ", input_batch_positional_embedding.shape
    )  # should be (4,3)
    input_embeddings = input_batch_token_embedding + input_batch_positional_embedding
    print(
        "Shape of Input Embedding for the current batch = ", input_embeddings.shape
    )  # should be (8,4,3)
    # multi_head_attention = MultiHeadAttention(
    #     num_heads=NUM_HEADS,
    #     context_length=CONTEXT_LENGTH,
    #     dim_in=MHA_DIM_IN,
    #     dim_out=MHA_DIM_OUT,
    # )
    # mha_out = multi_head_attention(input_embeddings)
    # print(
    #     "Shape of MHA Out = ", mha_out.shape
    # )  # should be (8,4,12) [BATCH_SIZE, CONTEXT_LENGTH, NUM_HEADS*MHA_DIM_OUT]

    multi_head_attention_efficient = MultiHeadAttentionEfficient(
        context_length=CONTEXT_LENGTH,
        dim_in=MHA_DIM_IN,
        dim_out=MHA_DIM_OUT,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
    )
    mha_efficient_out = multi_head_attention_efficient(input_embeddings)
    print(
        "Shape of MHA Shared Weights Out = ", mha_efficient_out.shape
    )  # should be (8,4,12) [BATCH_SIZE, CONTEXT_LENGTH, NUM_HEADS*MHA_DIM_OUT]
