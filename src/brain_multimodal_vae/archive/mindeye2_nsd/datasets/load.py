import webdataset as wds
import h5py
import torch

def my_split_by_node(urls): return urls

def load_all_subj_data(data_path, subj_list, data_range, subj_num_ssessions_list=None):
      all_subj_data = {}

      for s in subj_list:
          if data_range == "no-shared1000":
              data_url = f"{data_path}/wds/subj0{s}/train/" "{0.." + f"{subj_num_ssessions_list[s-1]-1}" + "}.tar"
          elif data_range == "shared1000":
              data_url = f"{data_path}/wds/subj0{s}/new_test/" + "0.tar"
          elif data_range == "all":
              data_url = (
                  f"{data_path}/wds/subj0{s}/train/" "{0.." + f"{subj_num_ssessions_list[s-1]-1}" + "}.tar::"
                  f"{data_path}/wds/subj0{s}/new_test/0.tar"
              )
          else:
              raise ValueError(f"Unsupported data_range: {data_range}")

          subj_iter_data = wds.WebDataset(data_url, resampled=False, shardshuffle=False, nodesplitter=my_split_by_node) \
                              .decode("torch") \
                              .rename(behav="behav.npy",
                                      past_behav="past_behav.npy",
                                      future_behav="future_behav.npy",
                                      olds_behav="olds_behav.npy") \
                              .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])

          # global_trial (behav[0, 5]) で昇順ソート
          subj_data = list(subj_iter_data)
          subj_data = sorted(subj_data, key=lambda sample: sample[0][0, 5].item())
          all_subj_data[f"subj0{s}"] = subj_data

      print("Loaded all subj data\n")
      
      return all_subj_data

def load_all_subj_voxels(data_path, subj_list):
    all_subj_voxels = {}
    all_subj_num_voxels = {}

    for subj in subj_list:
        s = f"{int(subj):02d}"
        f = h5py.File(f"{data_path}/betas_all_subj{s}_fp32_renorm.hdf5", "r")
        betas = torch.Tensor(f["betas"][:]).to("cpu")
        all_subj_voxels[f"subj{s}"] = betas
        all_subj_num_voxels[f"subj{s}"] = betas[0].shape[-1]

    print("Loaded all subj voxels\n")
    
    return all_subj_voxels, all_subj_num_voxels