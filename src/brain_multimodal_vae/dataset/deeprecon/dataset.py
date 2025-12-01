import torch
from torch.utils.data import Dataset
from itertools import product, zip_longest
from .data import get_label_set

class DeepReconDataset(Dataset):
    def __init__(self, brain_dict, label_dict, subject_list, group=False, set_mode="or", include_missing=False):
        self.brain_dict = brain_dict
        self.label_dict = label_dict
        self.subject_list = subject_list
        self.group = group
        self.set_mode = set_mode
        self.include_missing = include_missing
        self.idxs_dict_list = self.get_grouped_idxs_dict_list()
    
    def __getitem__(self, index):
        idx_dict = self.idxs_dict_list[index]

        brain_dict = {}
        for subject, brain in self.brain_dict.items():
            idx = idx_dict[subject]
            brain_dict[subject] = brain[idx] if idx != -1 else torch.full_like(brain[0], -1)

        label_dict = {}
        for subject, label in self.label_dict.items():
            idx = idx_dict[subject]
            label_dict[subject] = label[idx] if idx != -1 else torch.full_like(label[0], -1)

        return brain_dict, label_dict, idx_dict
    
    def __len__(self):
        return len(self.idxs_dict_list)

    def get_grouped_idxs_dict_list(self):
        label_set = get_label_set(self.label_dict, self.subject_list, self.set_mode)

        grouped_idx_dict_list = []

        for label in label_set:
            label_idx_list = []
            for s in self.subject_list:
                label_idx = torch.where(self.label_dict[f"{s}"] == label)[0]

                if self.include_missing & (label_idx.numel() == 0):
                    label_idx_list.append([None])
                else:
                    label_idx_list.append(label_idx)

            if self.group:
                it = product(*label_idx_list)
            elif not self.include_missing:
                it = zip(*label_idx_list)
            else:
                it = zip_longest(*label_idx_list)

            for combo in it:
                grouped_idx_dict = {}
                for s, idx in zip(self.subject_list, combo):
                    grouped_idx_dict[f"{s}"] = idx if idx is not None else torch.tensor(-1)

                grouped_idx_dict_list.append(grouped_idx_dict)

        return grouped_idx_dict_list