import os
import torch
import pandas as pd

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

def get_sub_dict(dict, key_list):
    return {k: dict[k] for k in key_list if k in dict}

def collate_batch_dict_list(batch_dict_list):
    collated_dict = {
        key: torch.cat([chunk[key] for chunk in batch_dict_list], dim=0)
        for key in batch_dict_list[0].keys()
    }
    
    return collated_dict

def save_df(df, save_dir, file_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df.to_csv(os.path.join(save_dir, file_name), index=None)