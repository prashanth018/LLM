from gpt import GPT
from utils.constants import CONFIG_GPT2_124M, GPT2_MODEL_DIRECTORY, GPT2_MODEL_SIZE
from utils.gpt_download import download_gpt2, load_gpt2
import torch


def download_124M():
    download_gpt2(GPT2_MODEL_SIZE, GPT2_MODEL_DIRECTORY)


def load_and_map_gpt2(config):
    settings, params = load_gpt2(GPT2_MODEL_SIZE, GPT2_MODEL_DIRECTORY)
    config["vocab_size"] = settings["n_vocab"]
    config["context_length"] = settings["n_ctx"]
    config["stride"] = settings["n_ctx"]
    config["n_heads"] = settings["n_head"]
    config["n_layers"] = settings["n_layer"]
    print(params.keys())
    model = GPT(config)
    t = lambda x: torch.tensor(x)
    with torch.no_grad():
        model.token_embedding.embedding.weight.copy_(t(params["wte"]))
        model.positional_embedding.embedding.weight.copy_(t(params["wpe"]))
        for i in range(len(params["blocks"])):
            block = params["blocks"][i]
            # layer 1
            model.transformers[i].norm1.scale.copy_(t(block["ln_1"]["g"]))
            model.transformers[i].norm1.shift.copy_(t(block["ln_1"]["b"]))
            # mha
            c_attn_w = block["attn"]["c_attn"]["w"]
            c_attn_b = block["attn"]["c_attn"]["b"]
            dim = model.transformers[i].mha.W_Q.weight.shape[0]
            # query
            model.transformers[i].mha.W_Q.weight.copy_(t(c_attn_w[:, :dim].T))
            model.transformers[i].mha.W_Q.bias.copy_(t(c_attn_b[:dim]))
            # key
            model.transformers[i].mha.W_K.weight.copy_(t(c_attn_w[:, dim : 2 * dim].T))
            model.transformers[i].mha.W_K.bias.copy_(t(c_attn_b[dim : 2 * dim]))
            # value
            model.transformers[i].mha.W_V.weight.copy_(t(c_attn_w[:, 2 * dim :].T))
            model.transformers[i].mha.W_V.bias.copy_(t(c_attn_b[2 * dim :]))
            # output
            model.transformers[i].mha.W_O.weight.copy_(
                t(block["attn"]["c_proj"]["w"].T)
            )
            model.transformers[i].mha.W_O.bias.copy_(t(block["attn"]["c_proj"]["b"]))
            # layer 2
            model.transformers[i].norm2.scale.copy_(t(block["ln_2"]["g"]))
            model.transformers[i].norm2.shift.copy_(t(block["ln_2"]["b"]))
            # ffn
            # expansion
            model.transformers[i].ffn.ffn[0].weight.copy_(
                t(block["mlp"]["c_fc"]["w"].T)
            )
            model.transformers[i].ffn.ffn[0].bias.copy_(t(block["mlp"]["c_fc"]["b"]))
            # contraction
            model.transformers[i].ffn.ffn[2].weight.copy_(
                t(block["mlp"]["c_proj"]["w"].T)
            )
            model.transformers[i].ffn.ffn[2].bias.copy_(t(block["mlp"]["c_proj"]["b"]))

        # final
        model.final_norm.scale.copy_(t(params["g"]))
        model.final_norm.shift.copy_(t(params["b"]))

        # out head
        model.out_head.weight.copy_(t(params["wte"]))
    return model


if __name__ == "__main__":
    # download_124M()
    # model = GPT(CONFIG_GPT2_124M)
    model = load_and_map_gpt2(CONFIG_GPT2_124M)

    # load_and_map_gpt2(model)
