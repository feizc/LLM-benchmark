import io 
import json 
import datasets
import pandas as pd 


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def make_supervised_data(
    tokenizer,
    train_path,
    val_path,
):
    return tokenizer 