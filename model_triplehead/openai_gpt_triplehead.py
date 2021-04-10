# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT model."""


import json
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers.file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_utils import SequenceSummary
from transformers.utils import logging
from transformers.models.openai.configuration_openai import OpenAIGPTConfig
from transformers import OpenAIGPTPreTrainedModel, OpenAIGPTModel


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "openai-gpt"
_CONFIG_FOR_DOC = "OpenAIGPTConfig"
_TOKENIZER_FOR_DOC = "OpenAIGPTTokenizer"

OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST = ["openai-gpt"]

@dataclass
class OpenAIGPTTripleHeadsModelOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not and predicting assigned personalities.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided):
            Language modeling loss.
        mc_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`mc_labels` is provided):
            Multiple choice classification loss for the response head.
        persona_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`persona_labels` is provided):
            Multiple choice classification loss for the persona head.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        mc_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_candidate_responses)`):
            Prediction scores of the multiple choice classification head for response (scores for each choice before SoftMax).
        persona_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_candidate_personas)`):
            Prediction scores of the multiple choice classification head for persona (scores for each choice before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    mc_loss: Optional[torch.FloatTensor] = None
    persona_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mc_logits: torch.FloatTensor = None
    persona_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


OPENAI_GPT_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.OpenAIGPTConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

OPENAI_GPT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.OpenAIGPTTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`__
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""

@add_start_docstrings(
    """
OpenAI GPT Model transformer with a language modeling head and 2 multiple-choice classification heads on top e.g. for
RocStories/SWAG tasks. The heads are linear layers. The language modeling head has its weights tied to the
input embeddings, the classification heads take as input the hidden state of a specified classification token index in the
input sequence).
""",
    OPENAI_GPT_START_DOCSTRING,
)
class OpenAIGPTTripleHeadsModel(OpenAIGPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        config.num_labels = 1
        self.transformer = OpenAIGPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.multiple_choice_head = SequenceSummary(config)
        self.persona_head = SequenceSummary(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=OpenAIGPTTripleHeadsModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        persona_token_ids=None,
        labels=None,
        mc_labels=None,
        persona_labels=None,
        num_candidate_personas=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        mc_token_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_candidate_responses)`, `optional`, default to index of the last token of the input):
            Index of the classification token in each input sequence. Selected in the range ``[0, input_ids.size(-1) - 1]``.
        persona_token_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_candidate_personas)`, `optional`, default to index of the last token of the input):
            Index of the classification token in each input sequence. Selected in the range ``[0, input_ids.size(-1) - 1]``.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-1, 0, ..., config.vocab_size - 1]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size - 1]``
        mc_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size)`, `optional`):
            Labels for computing the multiple choice classification loss for responses. Indices should be in ``[0, ...,
            num_candidate_responses - 1]`` where `num_candidate_responses` is the size of the second dimension of the input tensors - `num_candidate_personas` + 1. 
            (see `input_ids` above)
        persona_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size)`, `optional`):
            Labels for computing the multiple choice classification loss for personas. Indices should be in ``[0, ...,
            num_candidate_personas - 1]`` where `num_candidate_personas` is the size of the second dimension of the input tensors. (see
            `input_ids` above)

        Return:

        Examples::

            >>> from openai_gpt_tripleheads import OpenAIGPTTokenizer, OpenAIGPTTripleHeadsModel
            >>> import torch

            >>> tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
            >>> model = OpenAIGPTTripleHeadsModel.from_pretrained('openai-gpt')
            >>> tokenizer.add_special_tokens({'cls_token': '[CLS]'})  # Add a [CLS] to the vocabulary (we should train it also!)
            >>> model.resize_token_embeddings(len(tokenizer))

            >>> choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
            >>> input_ids = torch.tensor([tokenizer.encode(s) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
            >>> mc_token_ids = torch.tensor([input_ids.size(-1)-1, input_ids.size(-1)-1]).unsqueeze(0)  # Batch size 1

            >>> outputs = model(input_ids, mc_token_ids=mc_token_ids)
            >>> lm_logits = outputs.lm_logits
            >>> mc_logits = outputs.mc_logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # logits for language model head
        lm_logits = self.lm_head(hidden_states)

        # logits for multiple choice response head 
        # response head is applied to the first `num_candidate_responses - 1` hidden states and the last hidden state
        # (last hidden state represents gold response)
        num_candidate_responses = hidden_states.shape[1] - num_candidate_personas + 1
        mc_hidden_states = list(range(num_candidate_responses - 1)) + [-1]
        mc_logits = self.multiple_choice_head(hidden_states[:, mc_hidden_states], mc_token_ids[:, mc_hidden_states]).squeeze(-1)
        
        # logits for multiple choice persona head
        # response head is applied to the last `num_candidate_personas` hidden states
        # (last hidden state represents gold persona)
        persona_logits = self.persona_head(hidden_states[:, -num_candidate_personas:], persona_token_ids[:, -num_candidate_personas:]).squeeze(-1)
        
        # compare logits to labels for the gold candidates
        lm_loss, mc_loss, persona_loss = None, None, None
        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            mc_loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
        if persona_labels is not None:
            loss_fct = CrossEntropyLoss()
            persona_loss = loss_fct(persona_logits.view(-1, persona_logits.size(-1)), persona_labels.view(-1))
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits, mc_logits, persona_logits) + transformer_outputs[1:]
            if persona_loss is not None:
                output = (persona_loss,) + output
            if mc_loss is not None:
                output = (mc_loss,) + output
            return ((lm_loss,) + output) if lm_loss is not None else output

        return OpenAIGPTDoubleHeadsModelOutput(
            loss=lm_loss,
            mc_loss=mc_loss,
            persona_loss=persona_loss,
            logits=lm_logits,
            mc_logits=mc_logits,
            persona_logits=persona_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

