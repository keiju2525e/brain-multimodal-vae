import torch
from ..utils import collate_batch_dict_list

def get_prediction_dict(model, test_dl):
    result_brain_batch_dict_list = []
    result_label_batch_dict_list = []
    result_recon_batch_dict_list = []
    result_idx_batch_dict_list = []

    with torch.no_grad():
        for brain_batch_dict, label_batch_dict, idx_batch_dict in test_dl:
            for s in test_dl.dataset.subject_list:
                brain_batch_dict[f"{s}"] = brain_batch_dict[f"{s}"].to(torch.float32).to(model.device)
                label_batch_dict[f"{s}"] = label_batch_dict[f"{s}"].to(model.device)
                idx_batch_dict[f"{s}"] = idx_batch_dict[f"{s}"].to(model.device)

            result_brain_batch_dict_list.append(brain_batch_dict)
            result_recon_batch_dict_list.append(model.get_recon_dict(brain_batch_dict))
            result_label_batch_dict_list.append(label_batch_dict)
            result_idx_batch_dict_list.append(idx_batch_dict)

    result_brain_dict = collate_batch_dict_list(result_brain_batch_dict_list)
    result_recon_dict = collate_batch_dict_list(result_recon_batch_dict_list)
    result_label_dict = collate_batch_dict_list(result_label_batch_dict_list)
    result_idx_dict = collate_batch_dict_list(result_idx_batch_dict_list)

    return result_brain_dict, result_recon_dict, result_label_dict, result_idx_dict



