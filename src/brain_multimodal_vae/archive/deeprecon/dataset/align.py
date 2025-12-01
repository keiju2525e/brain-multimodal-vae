import numpy as np

def get_train_data(brain_data_dict, subj_list, rep, is_normalized=True):
    records = {}
    train_mean_dict = {}
    train_norm_dict = {}

    for subj in subj_list:
        s = f"{int(subj):02d}"

        subj_all_brain_data = brain_data_dict[f"subj{s}"].select("ROI_VC = 1")
        subj_image_indexes = brain_data_dict[f"subj{s}"].select("image_index").flatten()

        sorted_indexes = np.argsort(subj_image_indexes)
        subj_all_brain_data = subj_all_brain_data[sorted_indexes, :]
        subj_image_indexes = subj_image_indexes[sorted_indexes]

        mask_unit = np.zeros(5, dtype=bool)
        mask_unit[:rep] = True
        mask = np.tile(mask_unit, 1200)

        subj_all_brain_data = subj_all_brain_data[mask]
        subj_image_indexes = subj_image_indexes[mask]

        subj_all_brain_data_mean = np.mean(subj_all_brain_data, axis=0)[np.newaxis, :]
        subj_all_brain_data_norm = np.std(subj_all_brain_data, axis=0, ddof=1)[np.newaxis, :]

        train_mean_dict[f"subj{s}"] = subj_all_brain_data_mean
        train_norm_dict[f"subj{s}"] = subj_all_brain_data_norm

        if is_normalized:
            subj_all_brain_data = (subj_all_brain_data - subj_all_brain_data_mean) / subj_all_brain_data_norm
            
        seen = {}
        for image_index, brain_data in zip(subj_image_indexes, subj_all_brain_data):
            image_index_int = int(image_index)
            seen[image_index_int] = seen.get(image_index_int, 0) + 1
            occurrence = seen[image_index_int]

            key = (image_index_int, occurrence)
            entry = records.setdefault(key, {"image_index": image_index_int, "occurrence": occurrence})
            entry[f"subj{s}"] = brain_data

    train_data = [records[key] for key in sorted(records)]

    return train_data, train_mean_dict, train_norm_dict

def get_test_data(brain_data_dict, subj_list, train_mean_dict, train_norm_dict, is_normalized=True):
    records = {}

    for subj in subj_list:
        s = f"{int(subj):02d}"

        subj_all_brain_data = brain_data_dict[f"subj{s}"].select("ROI_VC = 1")
        subj_image_indexes = brain_data_dict[f"subj{s}"].select("image_index").flatten()

        sorted_indexes = np.argsort(subj_image_indexes)
        subj_all_brain_data = subj_all_brain_data[sorted_indexes, :]
        subj_image_indexes = subj_image_indexes[sorted_indexes]

        if is_normalized:
            subj_all_brain_data = (subj_all_brain_data - train_mean_dict[f"subj{s}"]) / train_norm_dict[f"subj{s}"]

        seen = {}
        for image_index, brain_data in zip(subj_image_indexes, subj_all_brain_data):
            image_index_int = int(image_index)
            seen[image_index_int] = seen.get(image_index_int, 0) + 1
            occurrence = seen[image_index_int]

            key = (image_index_int, occurrence)
            entry = records.setdefault(key, {"image_index": image_index_int, "occurrence": occurrence})
            entry[f"subj{s}"] = brain_data

    test_data = [records[key] for key in sorted(records)]

    return test_data