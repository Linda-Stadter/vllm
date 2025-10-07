# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2023 The IndicTrans2 Authors and AI4Bharat team. All rights reserved.
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
""" PyTorch IndicTrans model."""

import math
from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn

import vllm.engine.llm_engine
import vllm.inputs.preprocess
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (QKVCrossParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.tokenizer import AnyTokenizer

from .configuration_indictrans import IndicTransConfig
from .interfaces import SupportsV0Only


def custom_prepare_decoder_input_ids_for_generation(self, decoder_input_ids):
    """Custom decoder input preparation for IndicTrans models."""
    decoder_start_token_id = self.get_decoder_start_token_id()
    assert decoder_start_token_id is not None

    if decoder_input_ids is None:
        # For IndicTrans, we only need the decoder_start_token_id
        return [decoder_start_token_id]

    # If decoder_input_ids is provided, ensure it starts with decoder_start_token_id
    if (len(decoder_input_ids) == 0
            or decoder_input_ids[0] != decoder_start_token_id):
        decoder_input_ids = [decoder_start_token_id] + decoder_input_ids

    return decoder_input_ids


vllm.inputs.preprocess.InputPreprocessor._prepare_decoder_input_ids_for_generation = custom_prepare_decoder_input_ids_for_generation

from vllm.transformers_utils.detokenizer_utils import (
    _convert_tokens_to_string_with_added_encoders,
    convert_prompt_ids_to_tokens)


def custom_detokenize_incrementally(
        tokenizer: AnyTokenizer,
        all_input_ids: list[int],
        prev_tokens: Optional[list[str]],
        prefix_offset: int,
        read_offset: int,
        skip_special_tokens: bool = False,
        spaces_between_special_tokens: bool = True) -> str:

    tokenizer._switch_to_target_mode()

    new_token_id = all_input_ids[-1]
    # This is the first iteration for this sequence
    is_first_iter = prev_tokens is None
    if is_first_iter:
        (prev_tokens, prefix_offset,
         read_offset) = convert_prompt_ids_to_tokens(
             tokenizer,
             all_input_ids[:-1],
             skip_special_tokens=skip_special_tokens)
    assert prev_tokens is not None

    # If the new token id is out of bounds, return an empty string.
    if 0 <= new_token_id < tokenizer.tgt_vocab_size:
        # Put new_token_id in a list so skip_special_tokens is respected
        new_tokens = tokenizer.convert_ids_to_tokens(
            [new_token_id], skip_special_tokens=skip_special_tokens)
        if isinstance(new_tokens, str):
            new_tokens = [new_tokens]
    else:
        new_tokens = [""]
    output_tokens = prev_tokens + new_tokens

    # If this is the first iteration, return all tokens.
    if is_first_iter:
        new_tokens = output_tokens

    # The prefix text is necessary only to defeat cleanup algorithms in
    # the decode which decide to add a space or not depending on the
    # surrounding ids.
    if tokenizer.is_fast or not tokenizer.get_added_vocab():
        prefix_text = tokenizer.convert_tokens_to_string(
            output_tokens[prefix_offset:read_offset])
        new_text = tokenizer.convert_tokens_to_string(
            output_tokens[prefix_offset:])
    else:
        prefix_text = _convert_tokens_to_string_with_added_encoders(
            tokenizer,
            output_tokens[prefix_offset:read_offset],
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )
        new_text = _convert_tokens_to_string_with_added_encoders(
            tokenizer,
            output_tokens[prefix_offset:],
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )

    if len(new_text) <= len(prefix_text) or new_text.endswith("�"):
        # utf-8 char at the end means it's a potential unfinished byte sequence
        # from byte fallback tokenization.
        # If it's in the middle, it's probably a real invalid id generated
        # by the model
        tokenizer._switch_to_input_mode()
        return new_tokens, "", prefix_offset, read_offset

    new_text = new_text[len(prefix_text):]

    if hasattr(tokenizer, 'clean_up_tokenization_spaces'
               ) and tokenizer.clean_up_tokenization_spaces:
        new_text = tokenizer.clean_up_tokenization(new_text)

    tokenizer._switch_to_input_mode()
    return new_tokens, new_text, read_offset, len(output_tokens)


vllm.transformers_utils.detokenizer.detokenize_incrementally = custom_detokenize_incrementally


def create_position_ids_from_input_ids(input_ids, positions, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.
    """
    mask = input_ids.ne(padding_idx).int()
    return (positions + 1) * mask + padding_idx


class IndicTransSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self,
                 num_positions: int,
                 embedding_dim: int,
                 padding_idx: Optional[int] = None):
        super().__init__()
        self.offset = 2
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.make_weights(num_positions + self.offset, embedding_dim,
                          padding_idx)

    def make_weights(self,
                     num_embeddings: int,
                     embedding_dim: int,
                     padding_idx: Optional[int] = None):
        emb_weights = self.get_embedding(num_embeddings, embedding_dim,
                                         padding_idx)
        if hasattr(self, "weights"):
            # in forward put the weights on the correct dtype and device of the param
            emb_weights = emb_weights.to(dtype=self.weights.dtype,
                                         device=self.weights.device)

        self.register_buffer("weights", emb_weights, persistent=False)

    @staticmethod
    def get_embedding(num_embeddings: int,
                      embedding_dim: int,
                      padding_idx: Optional[int] = None):
        """
        Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly from the description in Section 3.5 of
        "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings,
                           dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)],
                        dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb.to(torch.get_default_dtype())

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        positions: torch.Tensor = None,
    ):
        # Create the position ids from the input token ids. Any padded tokens remain padded.
        position_ids = create_position_ids_from_input_ids(
            input_ids,
            positions,
            self.padding_idx,
        ).to(input_ids.device)

        # expand embeddings if needed
        max_pos = position_ids.max().item()
        if max_pos > self.weights.size(0):
            self.make_weights(max_pos + self.offset, self.embedding_dim,
                              self.padding_idx)
        return (self.weights.index_select(0, position_ids).detach())

    def create_position_ids_from_inputs_embeds(self, inputs_embeds,
                                               past_key_values_length):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1,
            sequence_length + self.padding_idx + 1,
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        return (position_ids.unsqueeze(0).expand(input_shape).contiguous() +
                past_key_values_length)


class IndicTransCrossAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        config: Optional[IndicTransConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.d_model = config.d_model
        self.embed_dim = embed_dim
        self.total_num_heads = num_heads
        self.total_num_kv_heads = self.total_num_heads
        self.head_dim = embed_dim // num_heads
        self.config = config
        self.prefix = prefix

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads "
                             f"(got `embed_dim`: {self.embed_dim}"
                             f" and `num_heads`: {num_heads}).")
        self.scaling = self.head_dim**-0.5

        # TP sharding sizes is accounted for within "*Parallel" layers.
        self.qkv_proj = QKVCrossParallelLinear(self.d_model,
                                               self.d_model //
                                               self.total_num_heads,
                                               self.total_num_heads,
                                               self.total_num_kv_heads,
                                               bias,
                                               quant_config=quant_config)

        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            quant_config=quant_config,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size

        if self.total_num_kv_heads >= tp_world_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_world_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_world_size % self.total_num_kv_heads == 0
        self.num_kv_heads = self.num_heads  # No GQA in bart
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn",
                              attn_type=AttentionType.ENCODER_DECODER)
        print(
            f"CrossAttention config: num_heads={self.num_heads}, head_dim={self.head_dim}, total_heads={self.total_num_heads}"
        )

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""
        q, k, v = self.qkv_proj(decoder_hidden_states, encoder_hidden_states)
        attn_output = self.attn(q, k, v)
        output, _ = self.out_proj(attn_output)
        return output


class IndicTransEncoderAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        config: Optional[IndicTransConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.d_model = config.d_model
        self.embed_dim = embed_dim
        self.total_num_heads = num_heads
        self.total_num_kv_heads = self.total_num_heads
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads "
                             f"(got `embed_dim`: {self.embed_dim}"
                             f" and `num_heads`: {num_heads}).")
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.d_model,
            self.d_model // self.total_num_heads,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
        )

        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            quant_config=quant_config,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size

        if self.total_num_kv_heads >= tp_world_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_world_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_world_size % self.total_num_kv_heads == 0
        self.num_kv_heads = self.num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn",
                              attn_type=AttentionType.ENCODER)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v)
        output, _ = self.out_proj(attn_output)
        return output


class IndicTransDecoderSelfAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        config: Optional[IndicTransConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.d_model = config.d_model
        self.embed_dim = embed_dim
        self.total_num_heads = num_heads
        self.total_num_kv_heads = self.total_num_heads
        self.head_dim = embed_dim // num_heads
        self.config = config
        self.prefix = prefix

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads "
                             f"(got `embed_dim`: {self.embed_dim}"
                             f" and `num_heads`: {num_heads}).")
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.d_model,
            self.d_model // self.total_num_heads,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
        )

        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            quant_config=quant_config,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size

        if self.total_num_kv_heads >= tp_world_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_world_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_world_size % self.total_num_kv_heads == 0
        self.num_kv_heads = self.num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn",
                              attn_type=AttentionType.DECODER)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v)
        output, _ = self.out_proj(attn_output)
        return output


class IndicTransEncoderLayer(nn.Module):

    def __init__(self, vllm_config: IndicTransConfig, prefix: str = ""):
        super().__init__()
        self.embed_dim = vllm_config.encoder_embed_dim
        self.self_attn = IndicTransEncoderAttention(
            embed_dim=self.embed_dim,
            num_heads=vllm_config.encoder_attention_heads,
            config=vllm_config,
            cache_config=None,
            quant_config=None,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation_fn = get_act_fn(vllm_config.activation_function)
        self.fc1 = nn.Linear(self.embed_dim, vllm_config.encoder_ffn_dim)
        self.fc2 = nn.Linear(vllm_config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.normalize_before = vllm_config.encoder_normalize_before

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states = self.self_attn(hidden_states=hidden_states)

        hidden_states = residual + hidden_states
        if not self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        if not self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
                torch.isinf(hidden_states).any()
                or torch.isnan(hidden_states).any()):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states,
                                        min=-clamp_value,
                                        max=clamp_value)

        return hidden_states


class IndicTransDecoderLayer(nn.Module):

    def __init__(self, vllm_config: IndicTransConfig, prefix: str = ""):
        super().__init__()
        self.embed_dim = vllm_config.decoder_embed_dim

        self.self_attn = IndicTransDecoderSelfAttention(
            embed_dim=self.embed_dim,
            num_heads=vllm_config.decoder_attention_heads,
            config=vllm_config,
            cache_config=None,
            quant_config=None,
            prefix=f"{prefix}.self_attn",
        )
        self.activation_fn = get_act_fn(vllm_config.activation_function)

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.encoder_attn = IndicTransCrossAttention(
            embed_dim=self.embed_dim,
            num_heads=vllm_config.decoder_attention_heads,
            config=vllm_config,
            cache_config=None,
            quant_config=None,
            prefix=f"{prefix}.encoder_attn",
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, vllm_config.decoder_ffn_dim)
        self.fc2 = nn.Linear(vllm_config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.normalize_before = vllm_config.decoder_normalize_before

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
        """
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(hidden_states=hidden_states)

        hidden_states = residual + hidden_states
        if not self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

        hidden_states = self.encoder_attn(
            decoder_hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states)

        hidden_states = residual + hidden_states
        if not self.normalize_before:
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        if not self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class IndicTransEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`IndicTransEncoderLayer`].

    Args:
        vllm_config: IndicTransConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self,
                 vllm_config: VllmConfig,
                 embed_tokens: Optional[nn.Embedding] = None,
                 prefix: str = ""):
        config = vllm_config.model_config.hf_config
        super().__init__()

        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.encoder_embed_dim
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(
            embed_dim) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.encoder_vocab_size, embed_dim,
                                         self.padding_idx)

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = IndicTransSinusoidalPositionalEmbedding(
            config.max_source_positions, embed_dim, self.padding_idx)

        self.layers = nn.ModuleList([
            IndicTransEncoderLayer(config,
                                   prefix=f"{prefix}.layers.{layer_idx}")
            for layer_idx in range(config.encoder_layers)
        ])
        self.layer_norm = (nn.LayerNorm(embed_dim)
                           if config.encoder_normalize_before else None)
        self.layernorm_embedding = (nn.LayerNorm(embed_dim)
                                    if config.layernorm_embedding else None)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
        """
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is None and inputs_embeds is None:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_ids, inputs_embeds, positions)
        embed_pos = embed_pos.to(inputs_embeds.device)

        hidden_states = inputs_embeds + embed_pos
        if self.layernorm_embedding is not None:
            hidden_states = self.layernorm_embedding(hidden_states)

        for _, encoder_layer in enumerate(self.layers):

            hidden_states = encoder_layer(hidden_states, )

        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)

        return hidden_states


class IndicTransDecoder(nn.Module):
    """
    Transformer decoder consisting of *vllm_config.decoder_layers* layers. Each layer is a [`IndicTransDecoderLayer`]

    Args:
        vllm_config: VllmConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self,
                 vllm_config: VllmConfig,
                 embed_tokens: Optional[nn.Embedding] = None,
                 prefix: str = ""):
        config = vllm_config.model_config.hf_config
        super().__init__()

        embed_dim = config.decoder_embed_dim
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_target_positions
        self.embed_scale = math.sqrt(
            embed_dim) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.decoder_vocab_size, embed_dim,
                                         self.padding_idx)

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = IndicTransSinusoidalPositionalEmbedding(
            config.max_target_positions, embed_dim, self.padding_idx)

        self.layers = nn.ModuleList(
            [IndicTransDecoderLayer(config, prefix=f"{prefix}.layers.{layer_idx}") \
              for layer_idx in range(config.decoder_layers)]
        )
        self.layer_norm = (nn.LayerNorm(embed_dim)
                           if config.decoder_normalize_before else None)
        self.layernorm_embedding = (nn.LayerNorm(embed_dim)
                                    if config.layernorm_embedding else None)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
        """

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is None and inputs_embeds is None:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # embed positions
        positions = self.embed_positions(input_ids, inputs_embeds, positions)
        positions = positions.to(inputs_embeds.device)

        hidden_states = inputs_embeds + positions
        if self.layernorm_embedding is not None:
            hidden_states = self.layernorm_embedding(hidden_states)

        for _, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )

        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)

        return hidden_states


class IndicTransModel(nn.Module):

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        self.encoder = IndicTransEncoder(vllm_config,
                                         prefix=f"{prefix}.encoder")
        self.decoder = IndicTransDecoder(vllm_config,
                                         prefix=f"{prefix}.decoder")

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
            self,
            encoder_input_ids: Optional[torch.LongTensor] = None,
            encoder_positions: Optional[torch.LongTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_positions: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            intermediate_tensors: Optional[
                IntermediateTensors] = None,  # Support IntermediateTensors
    ) -> torch.Tensor:

        encoder_outputs = None
        if encoder_input_ids.numel() > 0:
            encoder_outputs = self.encoder(
                input_ids=encoder_input_ids,
                positions=encoder_positions,
                inputs_embeds=inputs_embeds,
            )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            positions=decoder_positions,
            encoder_hidden_states=encoder_outputs,
            inputs_embeds=decoder_inputs_embeds,
        )

        return decoder_outputs


class IndicTransForConditionalGeneration(nn.Module, SupportsV0Only):
    base_model_prefix = "model"

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        super().__init__()
        self.model = IndicTransModel(vllm_config,
                                     prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(num_embeddings=config.decoder_vocab_size,
                                      embedding_dim=config.decoder_embed_dim,
                                      bias=False)
        self.logits_processor = LogitsProcessor(config.decoder_vocab_size,
                                                config.decoder_vocab_size)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        positions: torch.Tensor = None,
        encoder_input_ids: torch.Tensor = None,
        encoder_positions: torch.Tensor = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        encoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:

        outputs = self.model(
            encoder_input_ids=encoder_input_ids,
            encoder_positions=encoder_positions,
            decoder_input_ids=input_ids,
            decoder_positions=positions,
            inputs_embeds=encoder_inputs_embeds,
            decoder_inputs_embeds=inputs_embeds,
        )

        return outputs

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:

        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)

        return logits

    def _rename_stacked_param(
        self,
        name: str,
    ) -> tuple[str, Optional[str]]:
        for key, mapping in self.stacked_params_mapping.items():
            if key in name:
                name = name.replace(key, mapping["param_name"])
                return name, mapping["shard_id"]
        return name, None

    stacked_params_mapping = {
        "q_proj": {
            "param_name": "qkv_proj",
            "shard_id": "q",
        },
        "k_proj": {
            "param_name": "qkv_proj",
            "shard_id": "k",
        },
        "v_proj": {
            "param_name": "qkv_proj",
            "shard_id": "v",
        },
    }

    params_mapping = {
        "beta": "bias",
        "gamma": "weight",
        "LayerNorm": "layernorm",
    }

    def _rename_key(self, key: str):
        prefix = f"{self.base_model_prefix}."
        key = key[len(prefix):] if key.startswith(prefix) else key

        for src, dst in self.params_mapping.items():
            key = key.replace(src, dst)

        return key

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        weights_list = list(weights)

        for name, loaded_weight in weights_list:
            name, shard_id = self._rename_stacked_param(name)

            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                if shard_id:
                    weight_loader(param, loaded_weight, shard_id)
                else:
                    weight_loader(param, loaded_weight)

                loaded_params.add(name)

        return loaded_params
