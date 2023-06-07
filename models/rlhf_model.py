import torch 
import transformers 
from torch import Tensor, nn 
from typing import Dict, Optional, Callable 
import torch.nn.functional as F
from typing import Sequence, Union

import abc 
from .reward_model import get_transformer_hidden_size 


class Policy(nn.Module, abc.ABC):
    def __init__(
        self, args, base_model: transformers.PreTrainedModel, base_tokenizer: transformers.PreTrainedTokenizer
    ):
        super().__init__()
        self.args = args
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer

    @abc.abstractmethod
    def forward(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        responses: Tensor,
        temperature: Optional[float] = None,
    ) -> Dict[str, Tensor]:
        raise NotImplementedError

    def respond(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        temperature: Optional[float] = None,
        num_return_sequences=1,
    ) -> Dict[str, Tensor]:
        assert not self.training, "Policy must be in eval model for generation."
        return self._post_respond(self._respond(queries, query_attn_masks, temperature, num_return_sequences))

    @abc.abstractmethod
    def _respond(
        self, queries: Tensor, query_attn_masks: Tensor, temperature: Optional[float] = None, num_return_sequences=1
    ) -> Dict[str, Tensor]:
        raise NotImplementedError

    def _post_respond(self, respond_outputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return respond_outputs


def compute_logprobs(logits: Tensor, labels: Tensor, ignore_index: int) -> Tensor:
    """Compute per-token logprobs, zeroing out places with ignore_index (padding)."""
    return -F.cross_entropy(logits.permute(0, 2, 1), labels, reduction="none", ignore_index=ignore_index)


def pad(inputs: Tensor, target_size: Union[torch.Size, Sequence[int]], value=0.0, left=True):
    current_size = inputs.size()
    diffs = tuple(ti - ci for ti, ci in zip_(target_size, current_size))
    pad_params = []
    for diff in diffs:
        pad_params = ([diff, 0] if left else [0, diff]) + pad_params
    res = F.pad(inputs, pad=pad_params, value=value)
    return res


def right_pad(inputs: Tensor, target_size: Union[torch.Size, Sequence[int]], value=0.0):
    return pad(inputs=inputs, target_size=target_size, value=value, left=False)


def alleq(l: Sequence, f: Optional[Callable] = lambda x, y: x == y):
    """Check all arguments in a sequence are equal according to a given criterion.

    Args:
        f: A bi-variate boolean function.
        l: A list/tuple.

    Returns:
        True if everything is equal; otherwise False.
    """
    return all(f(l[0], li) for li in l[1:])


def zip_(*args: Sequence):
    """Assert sequences of same length before zipping."""
    if len(args) == 0:
        return []
    assert alleq(args, lambda x, y: len(x) == len(y))
    return zip(*args)


class AutoregressivePolicy(Policy):
    def forward(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        responses: Tensor,
        temperature: Optional[float] = None,
    ) -> Dict[str, Tensor]:
        # TODO(lxuechen): Refactor attention mask. Here query_attn_masks overrides padding-based attention mask.
        if temperature is None:
            temperature = self.args.temperature
        input_ids = torch.cat([queries, responses], dim=1)
        attention_mask = input_ids.ne(self.base_tokenizer.pad_token_id)
        attention_mask[:, : queries.size(1)] = query_attn_masks
        # Fix position id issues and ensure consistency with `respond` for GPT and OPT.
        inputs = self.base_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        outputs = self.base_model(**inputs, output_hidden_states=True)
        original_logits = outputs.logits[:, -self.args.response_len - 1 : -1]
        logits = original_logits / temperature
        labels = input_ids[:, -self.args.response_len :]
        logprobs = compute_logprobs(logits, labels, ignore_index=self.base_tokenizer.pad_token_id)
        entropies = -(logits.softmax(dim=-1) * logits.log_softmax(dim=-1)).sum(dim=-1)
        last_hidden_state = outputs.hidden_states[-1][:, -self.args.response_len - 1 : -1]
        return dict(
            original_logits=original_logits,
            logits=logits,
            logprobs=logprobs,
            entropies=entropies,
            last_hidden_state=last_hidden_state,
        )

    def _respond(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        temperature: Optional[float] = None,
        num_return_sequences=1,
    ) -> Dict[str, Tensor]:
        if temperature is None:
            temperature = self.args.temperature
        self.base_model = self.base_model.to(torch.device('cuda'))
        sequences = self.base_model.generate(
            inputs=queries,
            attention_mask=query_attn_masks,
            do_sample=True,
            max_new_tokens=self.args.response_len,
            pad_token_id=self.base_tokenizer.pad_token_id,
            top_p=1.0,
            top_k=0,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            synced_gpus=False, #True,
        )
        responses = right_pad(
            sequences[:, queries.size(1) :],
            target_size=(sequences.size(0), self.args.response_len),
            value=self.base_tokenizer.pad_token_id,
        )
        return dict(responses=responses)  # Size (bsz * num_return_sequences, response_len).


class Value(nn.Module, abc.ABC):
    def __init__(
        self, args, base_model: transformers.PreTrainedModel, base_tokenizer: transformers.PreTrainedTokenizer
    ):
        super().__init__()
        self.args = args
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        hidden_size = get_transformer_hidden_size(base_model)
        value_head = torch.nn.Linear(hidden_size, 1)
        value_head.weight.data.zero_()
        value_head.bias.data.zero_()
        self.value_head = value_head.to(next(base_model.parameters()).device)

    @abc.abstractmethod
    def forward(self, queries: Tensor, query_attn_masks: Tensor, responses: Tensor) -> Dict[str, Tensor]:
        raise NotImplementedError


class AutoregressiveValue(Value):
    def forward(self, queries: Tensor, query_attn_masks: Tensor, responses: Tensor) -> Dict[str, Tensor]:
        sequences = torch.cat([queries, responses], dim=1)
        sequence_attn_masks = sequences.ne(self.base_tokenizer.pad_token_id)

        inputs = self.base_model.prepare_inputs_for_generation(
            input_ids=sequences,
            attention_mask=sequence_attn_masks,
            use_cache=False,
        )
        outputs = self.base_model(**inputs, return_dict=True)
        # value[t]: \hat{V}(sequences_{:t-1}); must align with `_estimate_advantage`.
        #last_hidden_state = outputs.last_hidden_state[:, queries.size(1) - 1 : -1]
        #values = self.value_head(last_hidden_state).squeeze(-1)
        values = outputs.rewards 
        return dict(values=values)


class ActorCritic(nn.Module):
    def __init__(self, policy: Policy, value_model: Value):
        super(ActorCritic, self).__init__()
        self.policy = policy
        self.value_model = value_model
        self.value_model = self.value_model.to(torch.device('cuda'))

    def forward(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        responses: Tensor,
        temperature: Optional[float] = None,
    ) -> Dict[str, Tensor]:
        # Assume the policy and value model share the same tokenizer.
        o1 = self.policy(queries, query_attn_masks, responses, temperature)
        o2 = self.value_model(queries, query_attn_masks, responses)
        return {**o1, **o2}

    def respond(
        self, queries: Tensor, query_attn_masks: Tensor, temperature: Optional[float] = None
    ) -> Dict[str, Tensor]:
        return self.policy.respond(queries=queries, query_attn_masks=query_attn_masks, temperature=temperature)


def make_policy_with_base_model(
    args, base_model: transformers.PreTrainedModel, base_tokenizer: transformers.PreTrainedTokenizer
) -> Policy:
    if base_model.config.is_encoder_decoder:
        raise NotImplementedError
    else:
        return AutoregressivePolicy(args, base_model, base_tokenizer)


def make_value_with_base_model(
    args,
    base_model: transformers.PreTrainedModel,
    base_tokenizer: transformers.PreTrainedTokenizer,
) -> Value:
    if base_model.config.is_encoder_decoder:
        raise NotImplementedError
    else:
        return AutoregressiveValue(args, base_model, base_tokenizer)




