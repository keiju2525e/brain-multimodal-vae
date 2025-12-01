import os
import bdpy

def load_data(data_path, subj_list):
    train_brain_data_dict = {
          f"subj{int(subj):02d}": bdpy.BData(os.path.join(data_path, f"sub-{int(subj):02d}_NaturalImageTraining.h5"))
          for subj in subj_list
      }

    test_brain_data_dict = {
          f"subj{int(subj):02d}": bdpy.BData(os.path.join(data_path, f"sub-{int(subj):02d}_NaturalImageTest.h5"))
          for subj in subj_list
      }
    
    all_subj_num_voxels = {}

    for subj in subj_list:
        s = f"{int(subj):02d}"
        all_subj_num_voxels[f"subj{s}"] = train_brain_data_dict[f"subj{s}"].select("ROI_VC = 1").shape[1]

    return train_brain_data_dict, test_brain_data_dict, all_subj_num_voxels