import os 
import datasets
import pandas as pd 
from .utils import jload 
from .preprocess import SFTDataset, DataCollatorForSFTDataset
from .preprocess import BinaryRewardModelingDataset, DataCollatorForBinaryRewardModelingDataset, split_train_into_train_and_eval
from .preprocess import QueryResponseDataset, DataCollatorForQueryResponseDataset 



def make_supervised_data(
    tokenizer,
    data_path,
):
    prompt_dict = jload(os.path.join(data_path, 'prompt.json')) 
    train_data = datasets.load_dataset("json", data_files=os.path.join(data_path, 'sft.json'))
    eval_data = datasets.load_dataset("json", data_files=os.path.join(data_path, 'val.json')) 
    
    train_df = pd.concat([pd.DataFrame(train_data)]) 
    eval_df = pd.concat([pd.DataFrame(eval_data)]) 

    train_dataset = SFTDataset(
        df=train_df,
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
    )
    eval_dataset = SFTDataset(
        df=eval_df,
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
    )

    data_collator = DataCollatorForSFTDataset(tokenizer=tokenizer) 
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def make_binary_reward_modeling_data(
    tokenizer, 
    data_path,
): 
    prompt_dict = jload(os.path.join(data_path, 'prompt.json'))  
    human_preference = datasets.load_dataset("json", data_files=os.path.join(data_path, 'gpt4_preference.json')) 
    train_df = pd.DataFrame(human_preference)

    train_dataset = BinaryRewardModelingDataset(
        df=train_df,
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
        end_sequence_with_eos=False,
    )
    train_dataset, eval_dataset = split_train_into_train_and_eval(
        train_dataset=train_dataset,
        eval_size=500,
        seed=2023,
    )
    data_collator = DataCollatorForBinaryRewardModelingDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)



def make_rlhf_data(
    tokenizer,
    data_path,
):
    prompt_dict = jload(os.path.join(data_path, 'prompt.json'))   
    train_data = datasets.load_dataset("json", data_files=os.path.join(data_path, 'unlabeled.json'))
    eval_data = datasets.load_dataset("json", data_files=os.path.join(data_path, 'val.json')) 
    
    train_df = pd.concat([pd.DataFrame(train_data)]) 
    eval_df = pd.concat([pd.DataFrame(eval_data)]) 
    train_dataset = QueryResponseDataset(
        df=train_df,
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
        query_len=192,
    )
    eval_dataset = QueryResponseDataset(
        df=eval_df,
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
        query_len=192,
    )
    return dict(
        train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=DataCollatorForQueryResponseDataset()
    )

