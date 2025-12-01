import re
import torch
from tqdm import tqdm

from pixyz.distributions import Normal, ProductOfNormal
from pixyz.losses import Expectation as E
from pixyz.losses import KullbackLeibler, LogProb, Parameter
from pixyz.models import Model

from .distributions.networks import Encoder, Decoder
from .distributions.distributions import Inference, Generator

from .base import Base

class DMVAE(Base):
    def __init__(self, subject_list, n_voxels_dict, zp_dim, zs_dim, hidden_dim, optimizer_name, lr, weight_decay=0, decice="cpu", **kwargs):
        self.zp_dim = zp_dim
        self.zs_dim = zs_dim 

        super().__init__(subject_list, n_voxels_dict, hidden_dim, optimizer_name, lr, weight_decay, device)

    def get_network_dicts(self):
        enc_dict = {}
        dec_dict = {}

        for s in self.subject_list:
            sn = re.fullmatch(r"x(\d{2})", s).group(1)   

            enc_dict[f"zp__{s}"] = Encoder(x_dim=self.n_voxels_dict[f"{s}"], hidden_dim=self.hidden_dim)
            enc_dict[f"zs__{s}"] = Encoder(x_dim=self.n_voxels_dict[f"{s}"], hidden_dim=self.hidden_dim)
            dec_dict[f"{s}__zp{sn}_zs"] = Decoder(x_dim=self.n_voxels_dict[f"{s}"], hidden_dim=self.hidden_dim)

        return enc_dict, dec_dict

    def set_dist_dict(self):
        enc_dict, dec_dict = self.get_network_dicts()

        for s in self.subject_list:
            sn = re.fullmatch(r"x(\d{2})", s).group(1)

            # q_φ(zp_subj | x_subj): q_φ1(zp1 | x1), q_φ2(zp2 | x2), ...
            self.dist_dict[f"q_zp{sn}__{s}"] = Inference(enc=enc_dict[f"zp__{s}"], var=[f"zp{sn}"], cond_var=[f"{s}"], z_dim=self.zp_dim, hidden_dim=self.hidden_dim).to(self.device)
            
            # q_φ(zs_subj | x_subj): q_φ1(zs1 | x1), q_φ2(zs2 | x2), ...
            self.dist_dict[f"q_zs__{s}"] = Inference(enc=enc_dict[f"zs__{s}"], var=["zs"], cond_var=[f"{s}"], z_dim=self.zs_dim, hidden_dim=self.hidden_dim).to(self.device)

            # prior(zp_subj): prior(zp1), prior(zp2), ...
            self.dist_dict[f"prior_zp{sn}"] = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=[f"zp{sn}"], features_shape=[self.zp_dim], name="p_{prior}").to(self.device)

            # p_θ(x_subj | zp_subj, zs): p_θ1(x1 | zp1, zs), p_θ2(x2 | zp2, zs), ...
            self.dist_dict[f"p_{s}__zp{sn}_zs"] = Generator(dec=dec_dict[f"{s}__zp{sn}_zs"], var=[f"{s}"], cond_var=[f"zp{sn}", "zs"], z_dim=self.zp_dim+self.zs_dim, hidden_dim=self.hidden_dim).to(self.device)

        # q_φ(zs | x)
        self.dist_dict["q_zs__x"] = ProductOfNormal([self.dist_dict[f"q_zs__{s}"] for s in self.subject_list], name="q").to(self.device)

        # prior(zs)
        self.dist_dict["prior_zs"] = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["zs"], features_shape=[self.zs_dim], name="p_{prior}").to(self.device)

    def get_loss(self):
        loss = 0

        for s_t in self.subject_list:
            sn_t = re.fullmatch(r"x(\d{2})", s_t).group(1)

            loss_subj_target = 0

            joint_recon_loss = - E(self.dist_dict[f"q_zp{sn_t}__{s_t}"], E(self.dist_dict["q_zs__x"], LogProb(self.dist_dict[f"p_{s_t}__zp{sn_t}_zs"])))
            joint_recon_kl = KullbackLeibler(self.dist_dict[f"q_zp{sn_t}__{s_t}"], self.dist_dict[f"prior_zp{sn_t}"]) + KullbackLeibler(self.dist_dict[f"q_zs__x"], self.dist_dict[f"prior_zs"])
            loss_subj_target += joint_recon_loss + joint_recon_kl

            for s_s in self.subject_list:
                sn_s = re.fullmatch(r"x(\d{2})", s_s).group(1)

                if s_t == s_s:
                    self_recon_loss = - E(self.dist_dict[f"q_zp{sn_t}__{s_t}"], E(self.dist_dict[f"q_zs__{s_s}"], LogProb(self.dist_dict[f"p_{s_t}__zp{sn_t}_zs"])))
                    self_recon_kl = KullbackLeibler(self.dist_dict[f"q_zp{sn_t}__{s_t}"], self.dist_dict[f"prior_zp{sn_t}"]) + KullbackLeibler(self.dist_dict[f"q_zs__{s_s}"], self.dist_dict["prior_zs"])
                    loss_subj_target += self_recon_loss + self_recon_kl
                else:
                    cross_recon_loss = - E(self.dist_dict[f"q_zp{sn_t}__{s_t}"], E(self.dist_dict[f"q_zs__{s_s}"], LogProb(self.dist_dict[f"p_{s_t}__zp{sn_t}_zs"])))
                    cross_recon_kl = KullbackLeibler(self.dist_dict[f"q_zp{sn_t}__{s_t}"], self.dist_dict[f"prior_zp{sn_t}"]) + KullbackLeibler(self.dist_dict[f"q_zs__{s_s}"], self.dist_dict["prior_zs"])
                    loss_subj_target += cross_recon_loss + cross_recon_kl
                    
            loss += loss_subj_target
        
        return loss.mean()

    def get_recon_dict(self, x_dict):
        z_dict = {}
        recon_dict = {}

        with torch.no_grad():
            for s in self.subject_list:
                sn = re.fullmatch(r"x(\d{2})", s).group(1)

                z_dict[f"zp{sn}__{s}"] = self.dist_dict[f"q_zp{sn}__{s}"].sample(x_dict, return_all=False) 
                z_dict[f"prior_zp{sn}"] = self.dist_dict[f"prior_zp{sn}"].sample(batch_n=x_dict[f"x{s}"].size(0)) 
                z_dict[f"zs__{s}"] = self.dist_dict[f"q_zs__{s}"].sample(x_dict, return_all=False)
                
            z_dict["zs__x"] = self.dist_dict["q_zs__x"].sample(x_dict, return_all=False)

            for s_t in self.subject_list:
                sn_t = re.fullmatch(r"x(\d{2})", s_t).group(1)

                recon_dict[f"joint_recon_{s_t}"] = self.dist_dict[f"p_{s_t}__zp{sn_t}_zs"].sample_mean(z_dict[f"zp{sn_t}__{s_t}"] | z_dict[f"zs__x"])
                
                for s_s in self.subject_list:
                    if s_t == s_s:
                        recon_dict[f"self_recon_{s_t}"] = self.dist_dict[f"p_{s_t}__zp{sn_t}_zs"].sample_mean(z_dict[f"zp{sn_t}__{s_t}"] | z_dict[f"zs__{s_s}"])
                    else:
                        recon_dict[f"cross_recon_{s_t}__{s_s}"] = self.dist_dict[f"p_{s_t}__zp{sn_t}_zs"].sample_mean(z_dict[f"prior_zp{sn_t}"] | z_dict[f"zs__{s_s}"])
                        recon_dict[f"cross_recon_{s_t}__{s_t}_{s_s}"] = self.dist_dict[f"p_{s_t}__zp{sn_t}_zs"].sample_mean(z_dict[f"zp{sn_t}__{s_t}"] | z_dict[f"zs__{s_s}"])

        return recon_dict