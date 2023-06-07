from .rl_trainer import RLTrainer 
import os
from typing import Callable, Dict, Optional, Tuple

import accelerate
import pandas as pd
import torch
import tqdm
import transformers
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from transformers.modeling_utils import unwrap_model

from .reward_trainer import unpack_dict 
from inference import prepare_inputs 
from .rlhf_model import zip_ 
from .rl_trainer import merge_dict, jdump, makedirs
from utils import constants 


def whiten(values, shift_mean=True, epsilon=1e-8):
    assert values.size(0) >= 8, f"Internal error: Minibatch size {values.size(0)} is insufficient for whitening."
    mean, std = values.mean(), values.std(unbiased=False)  # noqa
    whitened = (values - mean) / (std + epsilon)
    if not shift_mean:
        whitened = whitened + mean
    return whitened



def flatten_dict(nested, sep=".", postprocess_fn=lambda *args: args):
    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, dict):  # collections.Mapping fails in py3.10.
                rec(v, prefix + k + sep, into)
            else:
                v = postprocess_fn(v)
                into[prefix + k] = v

    flat = {}
    rec(nested, "", flat)
    return flat


class RLHFTrainer(RLTrainer):
    def __init__(
        self,
        args,
        train_dataset,
        eval_dataset,
        data_collator: Callable,
        policy,
        ref_policy,
        reward_model,
        tokenizer: transformers.PreTrainedTokenizer,
        accelerator,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler=None,
    ):
        super(RLHFTrainer, self).__init__(
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            policy=policy,
            ref_policy=ref_policy,
            reward_model=reward_model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        from torch.utils.tensorboard import SummaryWriter  
        self.writer = SummaryWriter('./save/rlhf')

    def _shape_reward(
        self, rewards, responses, logprobs, ref_logprobs
    ):
        # For some reason, line below doesn't work.
        # kl = (logits.softmax(dim=-1) * (logits.log_softmax(dim=-1) - ref_logits.log_softmax(dim=-1))).sum(dim=-1)
        kl = torch.clamp(logprobs - ref_logprobs, min=0.0)
        non_score_rewards = -self.kl_ctl.value * kl
        shaped_rewards = non_score_rewards.clone()
        # This introduces a small index off by one bug if pad_token_id == eos_token_id.
        terminal_positions = (responses != self.tokenizer.pad_token_id).sum(dim=1) - 1
        shaped_rewards[list(range(rewards.size(0))), terminal_positions] += rewards
        return dict(shaped_rewards=shaped_rewards, non_score_rewards=non_score_rewards, kl=kl)

    def _estimate_advantage(self, rewards, values):
        """Generalized advantage estimation.

        Reference:
            https://arxiv.org/abs/1506.02438
        """
        #print(values.size())
        #rewards = rewards.unsqueeze(0)
        #values = values.unsqueeze(0) 
        if self.args.whiten_rewards:
            rewards = whiten(rewards, shift_mean=False)
        lastgaelam = 0
        advantages_reversed = []
        gen_length = self.args.response_len
        for t in reversed(range(gen_length)):
            nextvalues = values[t + 1] if t < gen_length - 1 else 0.0
            #print(nextvalues,) 
            #print(rewards[:, t]) 
            #print(values[:, t])
            delta = rewards[:, t] + self.args.gamma * nextvalues - values[t]
            lastgaelam = delta + self.args.gamma * self.args.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        # print(advantages.size(), values.size())
        returns = advantages + values.unsqueeze(1) 
        advantages = whiten(advantages, shift_mean=True)
        #print(returns.size(), advantages.size())
        #returns = returns.squeeze(0)
        #advantages = advantages.squeeze(0)
        return dict(returns=returns, advantages=advantages)

    #@torch.inference_mode()
    def rollout(self, queries_data):
        """Rollout trajectories with policy.

        Args:
            queries_data: Sequence of batches or DataLoader.
                Each batch is a dict with keys 'queries' and 'query_attn_masks'.

        Returns:
            Dictionary with keys
                'queries', 'query_attn_masks', 'responses',
                'logprobs', 'ref_logprobs', 'values',
                'rewards', 'non_score_rewards', 'shaped_rewards'.
        """
        # Give up dropout throughout.
        self.policy.eval()
        # self._make_fsdp_happy()
        # `keep_fp32_wrapper` retains the autocast wrapper of model.forward created by accelerate:
        #  recall one sets mixed precision options with accelerator.
        # The precise value of this arg doesn't matter here, since we use the unwrapped model only for respond.
        # Generally, try to use the wrapped model as much as you can, since it's got the autocast/cast-back wrappers.
        unwrapped_policy = self.policy  #self.accelerator.unwrap_model(self.policy, keep_fp32_wrapper=True)

        self.ref_policy.eval()
        self.reward_model.eval()

        rollouts = []
        for batch_idx, batch in tqdm.tqdm(
            enumerate(queries_data),
            # disable=not self.accelerator.is_main_process,
            desc="rollout",
        ):
            # Sample rollouts.
            queries, query_attn_masks = unpack_dict(
                prepare_inputs(batch, device=torch.device('cuda')), #device=self.accelerator.device),
                keys=("queries", "query_attn_masks"),
            ) 
            with torch.no_grad():
                respond_outputs = unwrapped_policy.respond(queries, query_attn_masks, temperature=self.args.temperature)
            (responses,) = unpack_dict(respond_outputs, ("responses",))

            # Evaluate logprobs of the samples.
            rollouts_batch = {"queries": queries, "query_attn_masks": query_attn_masks, "responses": responses}
            with torch.no_grad():
                policy_outputs = self.policy(**rollouts_batch, temperature=self.args.temperature)
                ref_policy_outputs = self.ref_policy(**rollouts_batch, temperature=self.args.temperature)
            
            policy_outputs = unpack_dict(
                policy_outputs, keys=("logprobs", "values", "entropies"), return_type=dict
            )
            ref_policy_outputs = unpack_dict(
                ref_policy_outputs, keys=("logprobs", "entropies"), return_type=dict
            )
            rollouts_batch.update(policy_outputs)
            rollouts_batch.update({f"ref_{key}": value for key, value in ref_policy_outputs.items()})

            # Evaluate reward of the samples.
            text_queries, text_responses = tuple(
                self.tokenizer.batch_decode(tensor, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for tensor in (queries, responses)
            )
            del queries, responses  # Prevent mistakes.

            # We retokenizer, since policy and reward model might not have the same tokenizer.
            # TODO(lxuechen): Avoid retokenization when policy and reward tokenizer are the same.
            text_sequences = [q + r for q, r in zip_(text_queries, text_responses)]
            # TODO(lxuechen): This response retokenization has issues with OPT, since the tokenizer always prepend
            #  <bos_token>. But the issue is local to post_reward, which isn't an issue if we don't penalize.
            sequences, responses = tuple(
                self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                for text in (text_sequences, text_responses)
            )
            sequences, responses = prepare_inputs((sequences, responses), device=torch.device('cuda'))
            with torch.no_grad():
                reward_outputs = self.reward_model(**sequences)
            
            reward_outputs = self.post_reward(reward_outputs, responses.input_ids)
            rollouts_batch.update(reward_outputs)

            # Shape reward with KL penalty.
            shape_reward_outputs = self._shape_reward(
                rewards=rollouts_batch["rewards"],
                responses=rollouts_batch["responses"],
                logprobs=rollouts_batch["logprobs"],
                ref_logprobs=rollouts_batch["ref_logprobs"],
            )
            rollouts_batch.update(shape_reward_outputs)

            rollouts_batch_cpu = {key: value.cpu() for key, value in rollouts_batch.items()}
            rollouts.append(rollouts_batch_cpu)

        # Items in dict need to be of same shape.
        rollouts = merge_dict(rollouts, merge_fn=torch.cat)
        # Estimating advantages outside the loop gives more samples for reward normalization.
        # print(rollouts)
        advantages = self._estimate_advantage(
            rewards=rollouts["shaped_rewards"].to(torch.device('cuda')),
            values=rollouts["values"].to(torch.device('cuda')),
        )
        advantages = {key: value.cpu() for key, value in advantages.items()}
        return {**rollouts, **advantages}

    def post_reward(self, reward_outputs, responses):
        """Assign bad reward values to sequences which didn't stop properly."""
        if self.args.truncate_token_ids is None:
            return reward_outputs

        def get_validity_mask(sequences, end_token_id: int):
            """Mark a batch element as False if the sequence doesn't end with `end_token_id` after `truncate_after`."""
            assert sequences.dim() == 2
            validity_mask = []
            for sequence in sequences:
                (nonzeros,) = (sequence == end_token_id).nonzero(as_tuple=True)
                if len(nonzeros) == 0:
                    validity_mask.append(False)
                else:
                    validity_mask.append(
                        self.args.truncate_after is None
                        or
                        # Last occurrence of `end_token_id` is after `truncate_after`.
                        nonzeros[-1] > self.args.truncate_after
                    )
            return torch.tensor(validity_mask, device=sequences.device)

        validity_masks = [get_validity_mask(responses, end_token_id) for end_token_id in self.args.truncate_token_ids]
        validity_mask = torch.stack(validity_masks).any(dim=0)  # Sequence is valid if it ends with any end token.
        rewards = reward_outputs["rewards"]
        rewards[~validity_mask] = self.args.penalty_reward_value
        return reward_outputs

    def compute_loss(self, rollouts):
        values, old_logprob, returns, advantages, queries, query_attn_masks, responses = prepare_inputs(
            unpack_dict(
                rollouts,
                keys=("values", "logprobs", "returns", "advantages", "queries", "query_attn_masks", "responses"),
            ),
            device=torch.device("cuda"), #self.accelerator.device,
        )
        self.policy.train() 
        outputs = self.policy(queries, query_attn_masks, responses, temperature=self.args.temperature)

        vpred = outputs["values"]
        vpredclipped = torch.clamp(
            vpred,
            min=values - self.args.cliprange_value,
            max=values + self.args.cliprange_value,
        ) 
        #print(vpred.size(), returns.size())
        vf_losses1 = (vpred.unsqueeze(1) - returns) ** 2.0
        vf_losses2 = (vpredclipped.unsqueeze(1) - returns) ** 2.0
        vf_loss = 0.5 * torch.maximum(vf_losses1, vf_losses2).mean()
        vf_clipfrac = (vf_losses2 > vf_losses1).to(torch.get_default_dtype()).mean()

        logprob = outputs["logprobs"]
        ratio = torch.exp(logprob - old_logprob)
        # When current policy is close to the old policy, the KL component of this advantage is approximately correct.
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, min=1.0 - self.args.cliprange, max=1.0 + self.args.cliprange)
        pg_loss = torch.maximum(pg_losses, pg_losses2).mean()
        pg_clipfrac = (pg_losses2 > pg_losses).to(torch.get_default_dtype()).mean()  # noqa

        loss = pg_loss + self.args.vf_coef * vf_loss

        entropy = outputs["entropies"].mean()
        approxkl = 0.5 * ((logprob - old_logprob) ** 2.0).mean()

        return_mean, return_var = returns.mean(), returns.var(unbiased=False)
        value_mean, value_var = values.mean(), values.var(unbiased=False)

        stats = dict(
            loss=dict(policy=pg_loss, value=vf_loss, total=loss),
            policy=dict(entropy=entropy, approxkl=approxkl, clipfrac=pg_clipfrac),
            returns=dict(mean=return_mean, var=return_var),
            val=dict(
                vpred=vpred.mean(),
                error=((vpred.unsqueeze(1) - returns) ** 2).mean(),
                clipfrac=vf_clipfrac,
                mean=value_mean,
                var=value_var,
            ),
        )
        # print(stats)
        return loss, flatten_dict(stats, sep="/", postprocess_fn=lambda x: x.detach())

    def record_step_stats(self, train_stats, rollouts, step_idx, **kwargs):
        kl = rollouts["kl"]
        kl_sum_seq, kl_avg_seq = kl.sum(dim=1).mean(dim=0), kl.mean()
        shaped_rewards = rollouts["shaped_rewards"].sum(dim=1).mean(dim=0)
        non_score_rewards = rollouts["non_score_rewards"].sum(dim=1).mean(dim=0)
        rewards = rollouts["rewards"].mean(dim=0)
        stats = {
            f"objective/kl_coef": kwargs["kl_coef"],
            f"objective/kl_sum_seq": kl_sum_seq,
            f"objective/kl_avg_seq": kl_avg_seq,
            f"objective/shaped_rewards": shaped_rewards,
            f"objective/non_score_rewards": non_score_rewards,
            f"objective/rewards": rewards,  # Original model reward.
            f"objective/lr": self.optimizer.param_groups[0]["lr"],
            f"objective/entropies": rollouts["entropies"].mean(),
            f"objective/ref_entropies": rollouts["ref_entropies"].mean(),
        }
        for k, v in train_stats.items():
            stats[f"ppo/{k}"] = v.mean(dim=0)
        stats = {key: value.item() if torch.is_tensor(value) else value for key, value in stats.items()}
        #if self.accelerator.is_main_process:
        #print(stats) 
        
        #self.log_metrics('train', stats) 
        for k, v in stats.items():
            k = k.split('/')[-1]
            self.writer.add_scalar(k, v, global_step=step_idx)

        if self.args.output_dir is not None:
            # Store rollout data to disk to debug.
            rollouts_to_disk = {
                key: self.tokenizer.batch_decode(
                    tensor, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                for key, tensor in unpack_dict(
                    rollouts, keys=("queries", "responses"), return_type=dict
                ).items()
            }
            rollouts_to_disk = pd.DataFrame(rollouts_to_disk).to_dict(orient="records")
            jdump(rollouts_to_disk, os.path.join(self.args.output_dir, "rollouts", f"step_{step_idx}.json"))
        return stats

    @torch.inference_mode()
    def save_model(self, output_dir: Optional[str] = None, give_rw_access=True, check_corrupted=True):
        # We don't use accelerator here because, we want to be frugal and only store the policy.
        # Moreover, we want easy loadability -- calling .from_pretrained on the folder. Full dump wouldn't allow this.

        # Logic:
        #   1. Retrieve the complete state dict of the wrapped model.
        #       (retrieving state dict of submodule can lead to loss of keys)
        #   2. Remove keys that are part of the value network.
        #   3. Rename keys that are part of the policy network, so that they match the naming standard.
        output_dir = self.args.output_dir if output_dir is None else output_dir
        makedirs(output_dir)

        model, tokenizer = self.policy, self.tokenizer
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        ):
            print("Gathering full state_dict...")
            state_dict = model.state_dict()
            print("Finished gathering full state_dict...")

        if self.accelerator.is_main_process:
            # Retain and remap policy keys.
            new_state_dict = dict()
            prefix = "policy.base_model."
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    new_state_dict[key[len(prefix) :]] = value
            state_dict = new_state_dict

            if check_corrupted:  # Let the checks run on GPU.
                is_corrupted = any(value.isnan().any().item() for value in state_dict.values())
                print(f"Is there nans in the state_dict to be dumped? {is_corrupted}")

            cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
            del state_dict

            unwrapped = unwrap_model(model).policy.base_model
            assert isinstance(
                unwrapped, (transformers.OPTForCausalLM, transformers.LlamaForCausalLM)
            ), f"Expected to save a generative policy, but found model to be of type: {type(unwrapped)}."
            if hasattr(unwrapped, "_keys_to_ignore_on_save"):
                print(f"keys to ignore on save: {unwrapped._keys_to_ignore_on_save}")
            print(f"Saving model checkpoint to {output_dir}") 
            unwrapped.save_pretrained(output_dir, state_dict=cpu_state_dict)

            tokenizer.save_pretrained(output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(self.args, os.path.join(output_dir, constants.TRAINING_ARGS_NAME))

            if give_rw_access:
                try:
                    os.system(f"chmod -R a+xwr {output_dir}")
                except Exception as e:
                    print(f"Failed to give read-write access to {output_dir}: {e}")


