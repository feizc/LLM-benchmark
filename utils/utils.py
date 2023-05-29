import io 
import json 
import transformers 
from typing import Callable, Optional, Sequence
import numpy as np 


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

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


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict 


def mean(*seqs):
    singleton = len(seqs) == 1
    means = [float(np.mean(seq)) for seq in seqs]
    return means[0] if singleton else means


def stable_resize_token_embeddings_and_tokenizer(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    special_tokens_dict: dict,
):
    """Resize tokenizer and embedding together.

    For new tokens, the embedding value is the average of all old embedding vectors.
    """
    tokenizer.add_special_tokens(special_tokens_dict)
    stable_resize_token_embeddings(model, len(tokenizer))


def stable_resize_token_embeddings(
    model: transformers.PreTrainedModel, target_size: int
):
    num_new_tokens = target_size - model.get_input_embeddings().weight.size(0)
    model.resize_token_embeddings(target_size)

    # New token embedding takes the average of existing ones.
    # We need this check since `num_new_tokens` can be negative if token embeddings were padded initially.
    # This can happen when there's multiple-of-64 padding.
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


