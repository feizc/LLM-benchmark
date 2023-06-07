import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import transformers 

from models import RewardConfig, RewardModel, RewardTrainer, compute_reward_modeling_metrics
from utils import make_binary_reward_modeling_data 
from utils import constants, stable_resize_token_embeddings_and_tokenizer
from dataclasses import dataclass, field 
from typing import List


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    pad_token: str = field(default=constants.DEFAULT_PAD_TOKEN)
    flash_attn: bool = field(default=False)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be left padded to this length always during training."
        },
    )
    label_names: List[str] = field(
        default_factory=lambda: ["index_0", "index_1", "choice"],
        metadata={
            "help": "Names of the labels in the dataset. "
            "This is needed to get transformers.Trainer to not throw those tensors away before `compute_loss`."
            "By default, the trainer throws away columns it doesn't recognize when creating the "
            "`train_dataloader` (see `_remove_unused_columns`). "
        },
    )
    padding: str = field(
        default="longest",
        metadata={
            "help": "Padding strategy. If 'max_length', pads to `model_max_length` always; this might lead to some "
            "redundant compute. If 'longest', pads to the longest sequence in the batch, capped by `model_max_length`."
        },
    )
    initialize_model_on_cpu: bool = field(
        default=False,
        metadata={
            "help": "Whether to initialize the model on CPU. "
            "If True, models on all processes will be first initialized on CPU; this is RAM-costly but faster."
        },
    )
    end_sequence_with_eos: bool = field(
        default=False,
        metadata={
            "help": "Whether to end sequences with EOS. "
            "Ending with EOS might help the reward model realize it's time to predict."
        },
    )
    resume_from_checkpoint: bool = field(default=False, metadata={"help": "If True, loads from last check point."})
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Use fast tokenizer if True. "
            "Fast LLaMA tokenizer forces protobuf downgrade to 3.20.3. "
            "Use fast tokenizer only if you can live with that."
        },
    )


def main(): 
    model_path = './ckpt' 
    data_path = './data'
    parser = transformers.HfArgumentParser(TrainingArguments) 
    training_args = parser.parse_args_into_dataclasses()[0] 
    print(training_args)

    config = RewardConfig.from_pretrained(pretrained_model_name_or_path=model_path)
    config.backbone_model_name_or_path = model_path 
    print(config)
    model = RewardModel(config=config,)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, 
        model_max_length=512,
        padding_side="right",
        padding="longest",
    ) 
    special_tokens_dict = dict(additional_special_tokens=[])
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = constants.DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = constants.DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = constants.DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = constants.DEFAULT_UNK_TOKEN
    stable_resize_token_embeddings_and_tokenizer(model.backbone_model, tokenizer, special_tokens_dict) 

    data_module = make_binary_reward_modeling_data(tokenizer, data_path) 
    training_args.report_to = ['tensorboard']

    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_reward_modeling_metrics,
        **data_module,
    )

    trainer.train() 
    trainer.evaluate() 

    trainer.save_model('reward')


if __name__ == "__main__":
    main() 
