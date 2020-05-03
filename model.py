import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss

from transformers import GPT2PreTrainedModel, GPT2Model


class GPT2LMEmojiModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.output_size, bias=False)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def tie_weights(self):
        return None

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if "past" in kwargs and kwargs["past"]:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        inputs = {"input_ids": input_ids}
        inputs.update(kwargs)
        return inputs

    def forward(
            self,
            input_ids=None,
            past=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            inputs_mask=None
    ):
        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,

            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits, hidden_states) + transformer_outputs[1:]
        if labels is not None and inputs_mask is not None:
            logits = lm_logits[inputs_mask].contiguous()

            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)

