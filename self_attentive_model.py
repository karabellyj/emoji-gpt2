import torch
import torch.nn as nn
from transformers import GPT2Tokenizer

from config import GPT2EmojiConfig
from model import GPT2LMEmojiModel

config_class, model_class, tokenizer_class = (GPT2EmojiConfig, GPT2LMEmojiModel, GPT2Tokenizer)


class SelfAttention(nn.Module):
    def __init__(self, input_size, att_unit, att_hops):
        super().__init__()
        self.ws1 = nn.Linear(input_size, att_unit, bias=False)
        self.ws2 = nn.Linear(att_unit, att_hops, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.tanh(self.ws1(x))
        out = self.ws2(out)

        att = self.softmax(out.permute(0, 2, 1))
        output = torch.bmm(att, out)
        return output, att


class SelfAttentiveEmojiGPT2(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.gpt_config = config_class.from_pretrained(config['model_path'])
        self.gpt_tokenizer = tokenizer_class.from_pretrained(config['model_path'])

        self.gpt = model_class.from_pretrained(config['model_path'], config=self.gpt_config)
        self.att_encoder = SelfAttention(self.gpt_config.output_size, config['attention-unit'],
                                         config['attention-hops'])
        self.fc = nn.Linear(self.gpt_config.output_size, config['nfc'])
        self.tanh = nn.Tanh()
        self.pred = nn.Linear(config['nfc'], config['classes'])
        self.drop = nn.Dropout(config['dropout'])

    def forward(self, inputs):
        out = self.gpt(inputs)[0]
        out, att = self.att_encoder(out)
        out = self.tanh(self.fc(self.drop(out.flatten())))
        preds = self.pred(self.drop(out))

        return preds, att


class SelfAttentiveEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.gpt_config = config_class.from_pretrained(config['model_path'])

        self.gpt = model_class.from_pretrained(config['model_path'], config=self.gpt_config)
        self.drop = nn.Dropout(config['dropout'])
        self.ws1 = nn.Linear(self.gpt_config.output_size, config['attention-unit'], bias=False)
        self.ws2 = nn.Linear(config['attention-unit'], config['attention-hops'], bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        self.tokenizer = tokenizer_class.from_pretrained(config['model_path'])
        self.attention_hops = config['attention-hops']

    def forward(self, inputs):
        output = self.gpt(inputs)[0]
        size = output.size()  # bsz, len, nemoj
        embeddings = output.view(-1, size[2])
        transformed_inputs = torch.transpose(inputs, 0, 1).contiguous()
        transformed_inputs = transformed_inputs.view(size[0], 1, size[1])
        concatenated_inp = torch.cat([transformed_inputs for _ in range(self.attention_hops)], 1)

        hbar = self.tanh(self.ws1(self.drop(embeddings)))  # bsz*len, attention-unit
        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # bsz, len, hop
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # bsz, hop, len
        penalized_alphas = alphas + (
                -10000 * (concatenated_inp == 0).float())
        # bsz, hop, len + bsz, hop, len
        alphas = self.softmax(penalized_alphas.view(-1, size[1]))  # [bsz*hop, len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, len]

        return torch.bmm(alphas, output), alphas


class Classifier(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.encoder = SelfAttentiveEncoder(config=config)
        self.fc = nn.Linear(self.encoder.gpt_config.output_size * config['attention-hops'], config['nfc'])

        self.drop = nn.Dropout(config['dropout'])
        self.tanh = nn.Tanh()
        self.pred = nn.Linear(config['nfc'], config['classes'])

    def forward(self, inputs):
        output, attention = self.encoder(inputs)
        output = output.view(output.size(0), -1)
        fc = self.tanh(self.fc(self.drop(output)))
        preds = self.pred(self.drop(fc))

        return preds, attention

    def encode(self, inputs):
        return self.encoder.forward(inputs)[0]
