import os
import re
import numpy as np
import torch
import bdpy

def load_data(data_dir, subject_list, n_train_repetitions=5, normalize=True, as_tensor=True):
    train_brain_dict = {}
    train_label_dict = {}
    test_brain_dict = {}
    test_label_dict = {}
    
    for s in subject_list:
        sn = re.fullmatch(r"x(\d{2})", s).group(1)

        train_data = bdpy.BData(os.path.join(data_dir, f"sub-{sn}_NaturalImageTraining.h5"))
        test_data = bdpy.BData(os.path.join(data_dir, f"sub-{sn}_NaturalImageTest.h5"))

        train_brain = train_data.select("ROI_VC")
        train_label = train_data.select("image_index").flatten().astype(int)
        test_brain = test_data.select("ROI_VC")
        test_label = test_data.select("image_index").flatten().astype(int)

        train_sort_idx = np.argsort(train_label, kind="stable")
        test_sort_idx = np.argsort(test_label, kind="stable")
        train_brain = train_brain[train_sort_idx]
        train_label = train_label[train_sort_idx]
        test_brain = test_brain[test_sort_idx]
        test_label = test_label[test_sort_idx]

        max_train_repetitions = 5
        train_mask_unit = np.zeros(max_train_repetitions, dtype=bool)
        train_mask_unit[:n_train_repetitions] = True
        train_mask = np.tile(train_mask_unit, 1200)

        train_brain = train_brain[train_mask]
        train_label = train_label[train_mask]

        if normalize:
            train_mean = np.mean(train_brain, axis=0)
            train_std = np.std(train_brain, axis=0, ddof=1)
            train_std = train_std + np.finfo(train_std.dtype).eps

            train_brain = (train_brain - train_mean) / train_std
            test_brain = (test_brain - train_mean) / train_std 
               
        train_brain_dict[f"{s}"] = torch.tensor(train_brain) if as_tensor else train_brain
        train_label_dict[f"{s}"] = torch.tensor(train_label) if as_tensor else train_label
        test_brain_dict[f"{s}"] = torch.tensor(test_brain) if as_tensor else test_brain
        test_label_dict[f"{s}"] = torch.tensor(test_label) if as_tensor else test_label

    return train_brain_dict, train_label_dict, test_brain_dict, test_label_dict

def select_train_data(train_brain_dict, train_label_dict, subject_list, n_shared_labels, n_unique_labels, generator=None):
    all_labels = torch.arange(1, 1201)

    shared_idx = torch.randperm(len(all_labels), generator=generator)[:n_shared_labels]
    shared_labels = all_labels[shared_idx]

    remaining_labels = all_labels[~torch.isin(all_labels, shared_labels)]

    for s in subject_list:
        unique_idx = torch.randperm(len(remaining_labels), generator=generator)[:n_unique_labels]
        unique_labels = remaining_labels[unique_idx]

        remaining_labels = remaining_labels[~torch.isin(remaining_labels, unique_labels)]

        selected_labels = torch.sort(torch.cat([shared_labels, unique_labels]))[0]

        select_mask = torch.isin(train_label_dict[f"{s}"], selected_labels)
        select_index = torch.where(select_mask)[0]
        train_brain_dict[f"{s}"] = train_brain_dict[f"{s}"][select_index]
        train_label_dict[f"{s}"] = train_label_dict[f"{s}"][select_index]

    return train_brain_dict, train_label_dict

def get_label_set(label_dict, subject_list, set_mode="or"):
    label_set_list = [set(label_dict[f"{s}"].tolist()) for s in subject_list]

    if set_mode == "or":
        return set.union(*label_set_list) if label_set_list else set()
    elif set_mode == "and":
        return set.intersection(*label_set_list) if label_set_list else set()