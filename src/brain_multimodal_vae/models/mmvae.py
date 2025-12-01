import torch
from tqdm import tqdm

from pixyz.distributions import Normal
from pixyz.losses import Expectation as E
from pixyz.losses import KullbackLeibler, LogProb, Parameter
from pixyz.models import Model

from .distributions.networks import Encoder, Decoder
from .distributions.distributions import Inference, Generator
from .distributions.moe import MixtureOfNormal

from .base import Base

class MMVAE(Model):
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
            # q_φ(z_subj | x_subj): q_φ1(z1 | x1), q_φ2(z2 | x2), ...
            self.dist_dict[f"q_z__{s}"] = Inference(enc=enc_dict[f"z__{s}"], var=[f"z"], cond_var=[f"{s}"], z_dim=self.z_dim, hidden_dim=self.hidden_dim).to(self.device)

            # p_θ(x_subj | z): p_θ1(x1 | z), p_θ2(x2 | z), ...
            self.dist_dict[f"p_{s}__z"] = Generator(dec=dec_dict[f"{s}__z"], var=[f"{s}"], cond_var=["z"], z_dim=self.z_dim, hidden_dim=self.hidden_dim).to(self.device)

        # q_φ(z | x)
        self.dist_dict["q_z__x"] = MixtureOfNormal([self.dist_dict[f"q_z__{s}"] for s in self.subject_list], name="q").to(self.device)

        # prior(z)
        self.dist_dict["prior_z"] = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["z"], features_shape=[self.z_dim], name="p_{prior}").to(self.device)

    def get_loss(self):
        recon_loss = - E(self.dist_dict[f"q_z__x"], sum([LogProb(self.dist_dict[f"p_{s}__z"]) for s in self.subject_list]))
        kl = sum([(1 / len(self.subject_list)) * KullbackLeibler(self.dist_dict[f"q_z__{s}"], self.dist_dict[f"prior_z"]) for s in self.subject_list])

        loss = recon_loss + kl

        return loss.mean()

    def get_recon_dict(self, x_dict):
        z_dict = {}
        recon_dict = {}

        with torch.no_grad():
            for s in self.subject_list:
                z_dict[f"z__{s}"] = self.dist_dict[f"q_z__{s}"].sample(x_dict, return_all=False) 
            
            z_dict["z__x"] = self.dist_dict["q_z__x"].sample(x_dict, return_all=False)

            for s_t in self.subject_list:
                recon_dict[f"joint_recon_{s_t}"] = self.dist_dict[f"p_{s_t}__z"].sample_mean(z_dict[f"z__x"])
                
                for s_s in self.subject_list:
                    if s_t == s_s:
                        recon_dict[f"self_recon_{s_t}"] = self.dist_dict[f"p_{s_t}__z"].sample_mean(z_dict[f"z__{s_s}"])
                    else:
                        recon_dict[f"cross_recon_{s_t}__{s_s}"] = self.dist_dict[f"p_{s_t}__z"].sample_mean(z_dict[f"z__{s_s}"])

        return recon_dict