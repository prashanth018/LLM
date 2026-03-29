from nltk.corpus import gutenberg
import re


class TokenizerV1:
    def __init__(self, corpus):
        self.unknown_token = "<UNK>"
        self.vocab = set(self.splitter(corpus))
        self.vocab.add(self.unknown_token)
        self.vocab_to_token = dict(zip(self.vocab, range(1, len(self.vocab) + 1)))
        self.token_to_vocab = dict(zip(range(1, len(self.vocab) + 1), self.vocab))

    def splitter(self, text):
        return re.split(r'([\s\[\]!"#$%&\'()*+,\-./:;<=>?@\\^_`{|}~])', text)

    def encode(self, text):
        input_text = self.splitter(text)
        tokens = [
            (
                self.vocab_to_token[w]
                if w in self.vocab_to_token
                else self.vocab_to_token[self.unknown_token]
            )
            for w in input_text
        ]
        return tokens

    def decode(self, tokens):
        output_text = [self.token_to_vocab[tok] for tok in tokens]
        return "".join(output_text)


if __name__ == "__main__":
    corpus = gutenberg.raw("shakespeare-caesar.txt")
    # li = corpus.split()
    # print(li[:30])
    tokenizer = TokenizerV1(corpus)
    text = "Thou art a Cobler, art thou?"
    print("Input Text = ", text)
    tokens = tokenizer.encode(text)
    print("Tokens = ", tokens)
    output_text = tokenizer.decode(tokens=tokens)
    print("Decoded Text = ", output_text)
