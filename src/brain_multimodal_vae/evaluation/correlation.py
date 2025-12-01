import torch
from itertools import permutations
from brain_multimodal_vae.dataset.deeprecon import get_label_set

def get_pattern_corr_result(subject_list, result_brain_dict, result_recon_dict, result_label_dict):
    shared_label_set = get_label_set(result_label_dict, subject_list, set_mode="and")
    shared_label = torch.tensor(list(shared_label_set))

    pattern_corr_result = []
    for s_t, s_s in permutations(subject_list, 2):
        target_brain = result_brain_dict[f"{s_t}"]
        target_label = result_label_dict[f"{s_t}"]

        recon_brain = result_recon_dict[f"cross_recon_{s_t}__{s_s}"]
        recon_label = result_label_dict[f"{s_s}"]

        pattern_corr_list = calc_pattern_corr(target_brain, target_label, recon_brain, recon_label, shared_label)

        for i, corr in enumerate(pattern_corr_list):
            pattern_corr_result.append({
                "subject_target": s_t,
                "subject_source": s_s,
                "correlation": corr.item(),
                "label" : i + 1
            })
    
    return pattern_corr_result

def calc_pattern_corr(target_brain, target_label, recon_brain, recon_label, shared_label):
    pattern_corr_list = []
    for l in shared_label:
        target_pattern = target_brain[(target_label == l), :]
        recon_pattern = recon_brain[(recon_label == l), :]

        corrs = torch.corrcoef(torch.cat([target_pattern, recon_pattern], dim=0))
        corr = corrs[:target_pattern.shape[0], recon_pattern.shape[0]:].mean()

        pattern_corr_list.append(corr)

    return pattern_corr_list

