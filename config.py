import emoji
from transformers import GPT2Config


class GPT2EmojiConfig(GPT2Config):
    def __init__(self, vocab_size=50257, n_positions=1024, n_ctx=1024, n_embd=768, n_layer=12, n_head=12,
                 resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1, layer_norm_epsilon=1e-5, initializer_range=0.02,
                 summary_type="cls_index", summary_use_proj=True, summary_activation=None, summary_proj_to_labels=True,
                 summary_first_dropout=0.1, **kwargs):
        super().__init__(vocab_size, n_positions, n_ctx, n_embd, n_layer, n_head, resid_pdrop, embd_pdrop, attn_pdrop,
                         layer_norm_epsilon, initializer_range, summary_type, summary_use_proj, summary_activation,
                         summary_proj_to_labels, summary_first_dropout, **kwargs)
        self.output_size = len(emoji.UNICODE_EMOJI.keys())
