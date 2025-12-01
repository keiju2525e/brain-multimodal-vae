import re
import torch
from tqdm import tqdm
from itertools import combinations

from pixyz.distributions import Normal, ProductOfNormal
from pixyz.losses import Expectation as E
from pixyz.losses import KullbackLeibler, LogProb, Parameter

from .distributions.networks import Encoder, Decoder
from .distributions.distributions import Inference, Generator

from .base import Base

class MVAE(Base):
    def __init__(self, subject_list, n_voxels_dict, z_dim, hidden_dim, optimizer_name, lr, weight_decay=0, device="cpu", **kwargs):
        self.z_dim = z_dim

        super().__init__(subject_list, n_voxels_dict, hidden_dim, optimizer_name, lr, weight_decay, device)

    def get_network_dicts(self):
        enc_dict = {}
        dec_dict = {}

        for s in self.subject_list:            
            enc_dict[f"z__{s}"] = Encoder(x_dim=self.n_voxels_dict[f"{s}"], hidden_dim=self.hidden_dim)
            dec_dict[f"{s}__z"] = Decoder(x_dim=self.n_voxels_dict[f"{s}"], hidden_dim=self.hidden_dim)

        return enc_dict, dec_dict

    def set_dist_dict(self):
        enc_dict, dec_dict = self.get_network_dicts()

        for s in self.subject_list:
            # q_φ(z_subj | x_subj): q_φ1(z | x1), q_φ2(z | x2), ...
            self.dist_dict[f"q_z__{s}"] = Inference(enc=enc_dict[f"z__{s}"], var=[f"z"], cond_var=[f"{s}"], z_dim=self.z_dim, hidden_dim=self.hidden_dim).to(self.device)

            # p_θ(x_subj | z): p_θ1(x1 | z), p_θ2(x2 | z), ...
            self.dist_dict[f"p_{s}__z"] = Generator(dec=dec_dict[f"{s}__z"], var=[f"{s}"], cond_var=["z"], z_dim=self.z_dim, hidden_dim=self.hidden_dim).to(self.device)

        # q_φ(z | x_subj1, x_subj2, ...): q_φ1(z | x1, x2), q_φ2(z | x1, x3), ...
        for n in range(2, len(self.subject_list) + 1):
            for combo in combinations(self.subject_list, n):
                self.dist_dict[f"q_z__{''.join(combo)}"] = ProductOfNormal([self.dist_dict[f"q_z__{s}"] for s in combo], name="q").to(self.device)
        
        # prior(z)
        self.dist_dict["prior_z"] = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["z"], features_shape=[self.z_dim], name="p_{prior}").to(self.device)

    def calc_dynamic_loss(self, input_dict, active_x_mask, **kwargs):
        loss = 0
        for n in range(1, len(self.subject_list) + 1):
            for combo in combinations(self.subject_list, n):
                joint_recon_loss = - E(self.dist_dict[f"q_z__{''.join(combo)}"], sum([LogProb(self.dist_dict[f"p_{s}__z"]) for s in combo]))
                joint_recon_kl = KullbackLeibler(self.dist_dict[f"q_z__{''.join(combo)}"], self.dist_dict[f"prior_z"])
                
                joint_x_mask = torch.tensor([1 if (s in combo) else 0 for s in self.subject_list]).to(self.device)
                joint_coef = (joint_x_mask == active_x_mask).all(dim=1).int()

                loss += (joint_coef * (joint_recon_loss + joint_recon_kl).eval(input_dict, **kwargs)).mean()

        for s_t in self.subject_list:
            for s_s in self.subject_list:
                if s_t == s_s:
                    self_recon_loss = - E(self.dist_dict[f"q_z__{s_s}"], LogProb(self.dist_dict[f"p_{s_t}__z"]))
                    self_kl = KullbackLeibler(self.dist_dict[f"q_z__{s_s}"], self.dist_dict["prior_z"])

                    self_x_mask = torch.tensor([1 if (s in [s_t, s_s]) else 0 for s in self.subject_list]).to(self.device)
                    self_coef = ((self_x_mask & active_x_mask) == self_x_mask).all(dim=1).int()

                    loss += (self_coef * (self_recon_loss + self_kl).eval(input_dict, **kwargs)).mean()
                else:
                    cross_recon_loss = - E(self.dist_dict[f"q_z__{s_s}"], LogProb(self.dist_dict[f"p_{s_t}__z"]))
                    cross_kl = KullbackLeibler(self.dist_dict[f"q_z__{s_s}"], self.dist_dict["prior_z"])

                    cross_x_mask = torch.tensor([1 if (s in [s_t, s_s]) else 0 for s in self.subject_list]).to(self.device)
                    cross_coef = ((cross_x_mask & active_x_mask) == cross_x_mask).all(dim=1).int()

                    loss += (cross_coef * (cross_recon_loss + cross_kl).eval(input_dict, **kwargs)).mean()

        return loss
    
    def get_static_loss_cls(self):
        loss = 0

        joint_recon_loss = - E(self.dist_dict[f"q_z__{''.join(self.subject_list)}"], sum([LogProb(self.dist_dict[f"p_{s}__z"]) for s in self.subject_list]))
        joint_kl = KullbackLeibler(self.dist_dict[f"q_z__{''.join(self.subject_list)}"], self.dist_dict[f"prior_z"])
        loss += (joint_recon_loss + joint_kl)

        for s_t in self.subject_list:
            for s_s in self.subject_list:
                if s_t == s_s:
                    self_recon_loss = - E(self.dist_dict[f"q_z__{s_s}"], LogProb(self.dist_dict[f"p_{s_t}__z"]))
                    self_kl = KullbackLeibler(self.dist_dict[f"q_z__{s_s}"], self.dist_dict["prior_z"])
                    loss += (self_recon_loss + self_kl)
                else:
                    cross_recon_loss = - E(self.dist_dict[f"q_z__{s_s}"], LogProb(self.dist_dict[f"p_{s_t}__z"]))
                    cross_kl = KullbackLeibler(self.dist_dict[f"q_z__{s_s}"], self.dist_dict["prior_z"])
                    loss += (cross_recon_loss + cross_kl)

        return loss.mean()
        
    def get_recon_dict(self, x_dict):
        z_dict = {}
        recon_dict = {}

        with torch.no_grad():
            for s in self.subject_list:
                z_dict[f"z__{s}"] = self.dist_dict[f"q_z__{s}"].sample(x_dict, return_all=False) 
            
            z_dict[f"z__{''.join(self.subject_list)}"] = self.dist_dict[f"q_z__{''.join(self.subject_list)}"].sample(x_dict, return_all=False)

            for s_t in self.subject_list:
                recon_dict[f"joint_recon_{s_t}"] = self.dist_dict[f"p_{s_t}__z"].sample_mean(z_dict[f"z__{''.join(self.subject_list)}"])
                
                for s_s in self.subject_list:
                    if s_t == s_s:
                        recon_dict[f"self_recon_{s_t}"] = self.dist_dict[f"p_{s_t}__z"].sample_mean(z_dict[f"z__{s_s}"])
                    else:
                        recon_dict[f"cross_recon_{s_t}__{s_s}"] = self.dist_dict[f"p_{s_t}__z"].sample_mean(z_dict[f"z__{s_s}"])

        return recon_dict