from torch.utils.data import Dataset
from nltk.corpus import gutenberg
import tiktoken


class GutenbergDataset(Dataset):
    def __init__(self, tokenizer, context_length, stride):
        self.context_length = context_length
        self.stride = stride
        self.tokenizer = tokenizer
        corpus = gutenberg.raw("shakespeare-caesar.txt")
        self.tokens = self.tokenizer.encode(corpus)
        self.input, self.target = self.create_data()

    def create_data(self):
        input = []
        target = []
        for i in range(0, len(self.tokens) - self.context_length, self.stride):
            input.append(self.tokens[i : i + self.context_length])
            target.append(self.tokens[i + 1 : i + self.context_length + 1])
        return (input, target)

    def __getitem__(self, key):
        return (self.input[key], self.target[key])

    def __len__(self):
        return len(self.input)


if __name__ == "__main__":
    # input = []
    # target = []
    # tokens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # context_length = 4
    # stride = 4
    # for i in range(0, len(tokens) - context_length, stride):
    #     input.append(tokens[i : i + context_length])
    #     target.append(tokens[i + 1 : i + context_length + 1])

    # for i in range(len(target)):
    #     print(input[i], target[i])
    dataset = GutenbergDataset(context_length=4, stride=4)
    print(dataset.__getitem__(4))
    print(dataset.__getitem__(5))
