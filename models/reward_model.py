import torch
import transformers
from torch import Tensor, nn
from transformers.utils.generic import ModelOutput 
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union


def get_transformer_hidden_size(model: transformers.PreTrainedModel):
    if isinstance(model, transformers.GPT2LMHeadModel):
        hidden_size_attr_name = "n_embd"
    elif isinstance(model, transformers.OPTForCausalLM):
        hidden_size_attr_name = "word_embed_proj_dim"
    elif isinstance(model, transformers.T5ForConditionalGeneration):
        hidden_size_attr_name = "d_model"
    else:
        # Hack to deal with the fact that transformers library changed the LLaMA model name.
        llama_cls = getattr(
            transformers, "LLaMAForCausalLM" if hasattr(transformers, "LLaMAForCausalLM") else "LlamaForCausalLM"
        )
        if isinstance(model, llama_cls):
            hidden_size_attr_name = "hidden_size"
        else:
            raise ValueError(f"Unknown base_model type: {type(model)}")
        from typing import Any, Mapping
    return getattr(model.config, hidden_size_attr_name)



class RewardConfig(transformers.PretrainedConfig): 
    model_type = "reward_model"

    # Huggingface doesn't allow non-kwargs for `__init__`.
    def __init__(self, pretrained_model_name_or_path=None, **kwargs):
        super(RewardConfig, self).__init__(**kwargs)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.backbone_model_name_or_path = pretrained_model_name_or_path 


class RewardModelOutput(ModelOutput):
    rewards: Tensor = None


class RewardModel(transformers.PreTrainedModel):
    config_class = RewardConfig

    def __init__(self, config: RewardConfig, **kwargs):
        super(RewardModel, self).__init__(config) 
        self.backbone_model = transformers.AutoModelForCausalLM.from_pretrained(config.backbone_model_name_or_path)
        hidden_size = get_transformer_hidden_size(self.backbone_model)
        reward_head = nn.Linear(hidden_size, 1)
        torch.nn.init.zeros_(reward_head.bias)
        self.reward_head = reward_head.to(next(self.backbone_model.parameters()).device)

    def forward(self, input_ids, attention_mask=None, return_dict=True, **kwargs):
        # We only compute the rewards and don't compute the logistic regression loss in this function so that it's
        # easier to use for later stages of reranking / RL training.
        #outputs = self.backbone_model.model(
        #    input_ids=input_ids, attention_mask=attention_mask, return_dict=True, **kwargs
        #) 
        outputs = self.backbone_model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True, **kwargs
        ) 
        last_hidden_state = outputs.last_hidden_state
        last_hidden_state_at_the_end = last_hidden_state[:, -1, :]
        rewards = self.reward_head(last_hidden_state_at_the_end).squeeze(-1)
        return RewardModelOutput(rewards=rewards) if return_dict else (rewards,)

