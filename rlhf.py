import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import transformers 
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForCausalLM 
from utils import constants 
from models import RewardModel, make_policy_with_base_model, make_value_with_base_model
from models import ActorCritic 
from utils import make_rlhf_data 
from typing import List, Optional 
import sys
import torch 
from models import RLHFTrainer 
import accelerate
from accelerate import DistributedDataParallelKwargs
from transformers.training_args import OptimizerNames

class MyAccelerator(accelerate.Accelerator):
    """Thin wrapper for accelerate.Accelerator."""

    def __repr__(self):
        return (
            f"  state={self.state}, \n"
            f"  gradient_accumulation_steps={self.gradient_accumulation_steps:.6f}, \n"
            f"  split_batches={self.split_batches}, \n"
            f"  step_scheduler_with_optimizer={self.step_scheduler_with_optimizer},\n"
            f")"
        )

    def unwrap_optimizer(self, optimizer: accelerate.accelerator.AcceleratedOptimizer):
        return optimizer.optimizer


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    wandb_project: str = field(default="rlhf")
    flash_attn: bool = field(default=False)
    optim: str = field(default=OptimizerNames.ADAMW_HF) #"adamw_torch")
    truncate_tokens: Optional[List[str]] = field(
        default_factory=lambda: None,
        metadata={
            "help": "Tokens in strings to truncate at first occurrence. "
            "This was used in original OAI summarization paper to avoid models returning incomplete sentences. "
        },
    )
    truncate_after: Optional[int] = field(
        default=None, metadata={"help": "Truncate after this number of tokens. Prevents early truncation."}
    )
    penalty_reward_value: float = field(
        default=-1.0,
        metadata={
            "help": "Reward assigned to sequences that are truncated, "
            "e.g., due to outputting incomplete sentences for given context window."
        },
    )
    total_epochs: int = field(default=10)
    rollout_batch_size: int = field(default=512)
    step_batch_size: int = field(default=256)
    rollout_accumulation_steps: int = field(default=16)
    rollout_per_device_batch_size: int = field(default=32)
    step_per_device_batch_size: int = field(default=2)
    noptepochs: int = field(default=2)
    vf_coef: float = field(default=0.1)
    cliprange: float = field(default=0.2)
    cliprange_value: float = field(default=0.2)
    gamma: float = field(default=1.0)
    lam: float = field(default=1.0)
    whiten_rewards: bool = field(default=True)
    adam_epsilon: float = field(
        default=1e-5,
        metadata={
            "help": "Epsilon for AdamW optimizer. "
            "This is the default for OAI PPO code and UW Quark code. "
            "This is not the Hugging Face default."
        },
    )
    temperature: float = field(default=1.0)
    kl_coef: float = field(default=0.2)
    target_kl: float = field(default=6.0)
    k_beta: float = field(default=0.1)
    adaptive_kl: bool = field(default=False)
    eval_batches: int = field(default=sys.maxsize, metadata={"help": "Maximum number of batches to evaluate on."})
    init_value_with_reward: bool = field(
        default=True, metadata={"help": "Initialize the value model with the reward model."}
    )
    save_steps_extra: Optional[str] = field(
        default=None,
        metadata={
            "help": "A list of predetermined checkpoints to save, represented in the format 'no1__no2__no3'. "
            "Parse this with str.split('__')."
        },
    )
    query_len: int = field(default=192)
    response_len: int = field(default=300)
    policy_model_name_or_path: str = field(default=None)
    reward_model_name_or_path: str = field(default=None)
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Use fast tokenizer if True. "
            "Fast LLaMA tokenizer forces protobuf downgrade to 3.20.3. "
            "Use fast tokenizer only if you can live with that."
        },
    )

    def __post_init__(self):
        # Super class' __post_init__ is very complicated; don't do super for now in case mess something up.
        # super().__post_init__()

        if self.tf32:  # super().__post_init__() actually does this.
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True  # noqa

        if self.save_steps_extra is not None:
            self.save_steps_extra_list = [int(string) for string in self.save_steps_extra.split("__")]
        else:
            self.save_steps_extra_list = []

    def set_truncate_token_ids(self, tokenizer: transformers.PreTrainedTokenizer):
        """Convert truncation token to token ids.

        This is called in RLTrainer.
        """
        truncate_tokens = self.truncate_tokens
        if truncate_tokens is None:
            truncate_token_ids = None
        else:
            truncate_token_ids = tokenizer.convert_tokens_to_ids(truncate_tokens)
        self.truncate_token_ids = truncate_token_ids


def main(): 
    policy_ckpt_path = './sft' 
    reward_ckpt_path = './reward'
    data_path = './data'
    parser = transformers.HfArgumentParser(TrainingArguments) 
    training_args = parser.parse_args_into_dataclasses()[0] 
    print(training_args)

    tokenizer = AutoTokenizer.from_pretrained(policy_ckpt_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(dict(pad_token=constants.DEFAULT_PAD_TOKEN)) 
    
    policy = make_policy_with_base_model(training_args, AutoModelForCausalLM.from_pretrained(policy_ckpt_path), tokenizer) 
    value_model = make_value_with_base_model(training_args, RewardModel.from_pretrained(reward_ckpt_path), tokenizer) 
    policy.train() 
    value_model.train()
    actor_critic = ActorCritic(policy=policy, value_model=value_model) 

    ref_policy = make_policy_with_base_model(training_args, AutoModelForCausalLM.from_pretrained(policy_ckpt_path), tokenizer) 
    ref_policy = ref_policy.to(torch.device('cuda'))
    ref_policy.requires_grad_(False) 

    reward_model = RewardModel.from_pretrained(reward_ckpt_path)  
    reward_model = reward_model.to(torch.device('cuda'))
    reward_model.requires_grad_(False) 
    model_module = dict(policy=actor_critic, ref_policy=ref_policy, reward_model=reward_model)

    data_module = make_rlhf_data(
        tokenizer=tokenizer, 
        data_path=data_path,
    )
    
    """
    accelerator = MyAccelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        log_with=["tensorboard"],
        logging_dir='./save',
        # even_batches=True,  # Make sure the batch size on each device is the same.
        split_batches=False,  # Don't break a batch into smaller chunks.
        step_scheduler_with_optimizer=False,  # Untie optimizer and scheduler step.
        # Value model might not use all parameters (e.g., lm-head) in the forward pass.
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    accelerator.init_trackers(
        training_args.wandb_project,
        init_kwargs={"wandb": {"name": training_args.run_name}},
        config=training_args.__dict__,
    )
    """
    training_args.report_to = ['tensorboard']
    trainer = RLHFTrainer(
        args=training_args,
        accelerator=None,
        **data_module,
        **model_module,
        tokenizer=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main() 

