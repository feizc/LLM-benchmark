import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from transformers import AutoTokenizer, AutoModelForCausalLM 
from utils import make_supervised_data, constants, stable_resize_token_embeddings_and_tokenizer
from dataclasses import dataclass, field
import transformers 
from transformers import Trainer 


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    pad_token: str = field(default=constants.DEFAULT_PAD_TOKEN)
    flash_attn: bool = field(default=False)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded to this length (and possibly truncated)."
            "Enforcing a consistent max length ensures memory usage is constant and predictable."
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
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        model_max_length=512,
        padding_side="right",
        padding="longest",
    )
    parser = transformers.HfArgumentParser(TrainingArguments) 
    training_args = parser.parse_args_into_dataclasses()[0]
    print(training_args)

    model = AutoModelForCausalLM.from_pretrained(model_path) 
    # Collect special tokens. Only add if non-existent.
    special_tokens_dict = dict(additional_special_tokens=[])
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = constants.DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = constants.DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = constants.DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = constants.DEFAULT_UNK_TOKEN
    stable_resize_token_embeddings_and_tokenizer(model, tokenizer, special_tokens_dict) 
    print(tokenizer.pad_token)
    
    data_module = make_supervised_data(tokenizer, data_path) 
    print(data_module) 
    training_args.report_to = ['tensorboard']
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )
    trainer.train() 
    trainer.save_model('sft') 

if __name__ == "__main__":
    main() 
