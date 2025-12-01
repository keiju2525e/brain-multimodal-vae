import pandas as pd
import numpy as np
import re
import torch

def load_things_fmri_data(things_fmri_dir, things_dir, subject_list, roi_list, label_mode="image", normalize=True, as_tensor=True):
    concept_to_idx_df = pd.read_csv(things_dir + "concept_to_idx.csv")
    image_to_idx_df = pd.read_csv(things_dir + "image_to_idx.csv")
    concept_to_idx_dict = pd.Series(concept_to_idx_df["concept_idx"].values, index=concept_to_idx_df["concept_name"]).to_dict()
    image_to_idx_dict = pd.Series(image_to_idx_df["image_idx"].values, index=image_to_idx_df["image_name"]).to_dict()
    
    train_brain_dict = {}
    train_label_dict = {}
    test_brain_dict = {}
    test_label_dict = {}
    for s in subject_list:
        sn = re.fullmatch(r"x(\d{2})", s).group(1)

        beta_df = pd.read_hdf(things_fmri_dir + f"sub-{sn}_ResponseData.h5").drop(columns=["voxel_id"])
        roi_df = pd.read_csv(things_fmri_dir + f"sub-{sn}_VoxelMetadata.csv")
        stim_df = pd.read_csv(things_fmri_dir + f"sub-{sn}_StimulusMetadata.csv")

        is_roi = roi_df[roi_list].any(axis=1)
        beta_df = beta_df[is_roi]
        beta_arr = beta_df.to_numpy().T

        if label_mode == "image":
            names = stim_df["stimulus"].str.replace(r'\.[^.]*$', '', regex=True)
            label_series = names.map(image_to_idx_dict)
        elif label_mode == "concept":
            names = stim_df["concept"].str.replace(r'\.[^.]*$', '', regex=True)
            label_series = names.map(concept_to_idx_dict)

        is_train = stim_df["trial_type"] == "train"
        is_test  = stim_df["trial_type"] == "test"

        train_label = label_series[is_train].to_numpy()
        test_label = label_series[is_test].to_numpy()
        train_brain = beta_arr[is_train.to_numpy()]
        test_brain  = beta_arr[is_test.to_numpy()]

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