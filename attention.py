import math
import types
from typing import Optional, Tuple
import pickle
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import numpy as np

def llama_new_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
    )

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(
            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
        )

    ### DuCAR's modification
    if hasattr(self, "use_attn"):
        use_attn = self.use_attn
        img_start_idx = self.img_start_idx
        img_end_idx = self.img_end_idx
    else:
        use_attn = False

    if hasattr(self, "use_cfg"):
        use_cfg = self.use_cfg
    else:
        use_cfg = False

    cls_sample_tokens = self.shared_dict['cls_sample_tokens'] + self.img_start_idx

    if use_attn and not use_cfg:
        if self.layer_idx == self.intermediate_layer_id and attn_weights.size(-1) == self.prompt_end_idx:
            img_attn_weights = attn_weights[:, :, img_end_idx + 1:self.prompt_end_idx, img_start_idx:img_end_idx]
            img_attn_weights = nn.functional.softmax(img_attn_weights, dim=-1, dtype=torch.float32)
            max_attn_weights = img_attn_weights.max(dim=1)[0]
            importance_scores = max_attn_weights.mean(dim=1)

            self.shared_dict['importance_scores'] = importance_scores

            sorted_scores, sorted_indices = torch.sort(importance_scores, dim=-1, descending=True)
            cumulative_scores = torch.cumsum(sorted_scores / sorted_scores.sum(dim=-1, keepdim=True), dim=-1)
            threshold_mask = cumulative_scores > self.threshold_ratio
            threshold_indices = torch.argmax(threshold_mask.int(), dim=-1, keepdim=True)
            topk_indices = sorted_indices[:, :threshold_indices.max() + 1]
            text_sample_tokens = topk_indices + self.img_start_idx
            text_sample_tokens = text_sample_tokens.flatten()
            self.shared_dict['text_sample_tokens'] = text_sample_tokens

            cls_sample_set = set(cls_sample_tokens.tolist())
            text_sample_set = set(text_sample_tokens.tolist())

            dual_tokens = torch.tensor(list(cls_sample_set & text_sample_set), dtype=torch.long,
                                              device=attn_weights.device)
            text_only = torch.tensor(list(text_sample_set - cls_sample_set), dtype=torch.long,
                                             device=attn_weights.device)
            visual_only = torch.tensor(list(cls_sample_set - text_sample_set), dtype=torch.long,
                                       device=attn_weights.device)

            self.shared_dict['dual_tokens'] = dual_tokens
            print("len(dual_tokens): ", len(dual_tokens))
            self.shared_dict['text_only'] = text_only
            print("len(text_only): ", len(text_only))
            self.shared_dict['visual_only'] = visual_only
            print("len(visual_only): ", len(visual_only))
            print("total_tokens: ", len(dual_tokens) + len(text_only) + len(visual_only))

            overlapping_indices = dual_tokens - self.img_start_idx
            importance = importance_scores[:, overlapping_indices]
            sigmoid_importance = torch.sigmoid(importance) * self.beta + self.alpha
            self.shared_dict['sigmoid_importance'] = sigmoid_importance.to(attn_weights.dtype)
            print("self.shared_dict['sigmoid_importance']: ", self.shared_dict['sigmoid_importance'])


        # Apply enhancement in layers >= intermediate_layer_id
        if hasattr(self, 'intermediate_layer_id') and self.layer_idx >= self.intermediate_layer_id:
            if 'dual_tokens' in self.shared_dict and len(self.shared_dict['dual_tokens']) > 0:
                attn_weights[:, :, -1, self.shared_dict['dual_tokens']] = (
                        attn_weights[:, :, -1, self.shared_dict['dual_tokens']].abs() * self.shared_dict['sigmoid_importance']
                        + attn_weights[:, :, -1, self.shared_dict['dual_tokens']]
                )

            if 'text_only' in self.shared_dict and len(self.shared_dict['text_only']) > 0:
                attn_weights[:, :, -1, self.shared_dict['text_only']] = (
                        attn_weights[:, :, -1, self.shared_dict['text_only']].abs() * self.alpha
                        + attn_weights[:, :, -1, self.shared_dict['text_only']]
                )
            if 'visual_only' in self.shared_dict and len(self.shared_dict['visual_only']) > 0:
                attn_weights[:, :, -1, self.shared_dict['visual_only']] = (
                        attn_weights[:, :, -1, self.shared_dict['visual_only']].abs() * self.alpha
                        + attn_weights[:, :, -1, self.shared_dict['visual_only']]
                )


    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )

    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def llama_modify(model, start_layer, end_layer, use_attn, alpha, beta, use_cfg,
                 img_start_idx, img_end_idx, prompt_end_idx, threshold_ratio):

    shared_dict = {'cls_sample_tokens': None, 'text_sample_tokens': None}
    model.model.vision_tower.shared_dict = shared_dict

    for i in range(start_layer, end_layer):
        model.model.layers[i].self_attn.shared_dict = shared_dict
        model.model.layers[i].self_attn.use_attn = use_attn
        model.model.layers[i].self_attn.alpha = alpha
        model.model.layers[i].self_attn.beta = beta
        model.model.layers[i].self_attn.use_cfg = use_cfg
        model.model.layers[i].self_attn.img_start_idx = img_start_idx
        model.model.layers[i].self_attn.img_end_idx = img_end_idx
        model.model.layers[i].self_attn.prompt_end_idx = prompt_end_idx
        model.model.layers[i].self_attn.intermediate_layer_id = start_layer
        model.model.layers[i].self_attn.threshold_ratio = threshold_ratio
        model.model.layers[i].self_attn.forward = types.MethodType(llama_new_forward, model.model.layers[i].self_attn)
