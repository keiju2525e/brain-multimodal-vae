import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from pixyz.models import Model

from abc import abstractmethod

class Base(Model):
    def __init__(self, subject_list, n_voxels_dict, hidden_dim, optimizer_name, lr, weight_decay=0, device="cpu", **kwargs):
        self.subject_list = subject_list
        self.n_voxels_dict = n_voxels_dict

        self.hidden_dim = hidden_dim

        self.device = device

        self.dist_dict = {}
        self.set_dist_dict()

        loss = self.get_static_loss_cls()

        self.optimizer_name = optimizer_name
        optimizer = self.get_optimizer()

        super().__init__(loss=loss, distributions=list(self.dist_dict.values()), optimizer=optimizer, optimizer_params={"lr": lr, "weight_decay": weight_decay})

    def get_optimizer(self):
        optimizers = {
            'rmsprop': optim.RMSprop,
            'adadelta': optim.Adadelta,
            'adagrad': optim.Adagrad,
            'adam': optim.Adam
        }

        return optimizers[self.optimizer_name]

    # override "https://github.com/masa-su/pixyz/blob/v0.3.3/pixyz/models/model.py"
    def train(self, train_input_dict={}, include_missing=True, active_x_mask=None, **kwargs):
        self.distributions.train()

        self.optimizer.zero_grad()

        if include_missing:
            loss = self.calc_dynamic_loss(train_input_dict, active_x_mask, **kwargs)
        else:
            loss = self.loss_cls.eval(train_input_dict, **kwargs)

        loss.backward()

        if self.clip_norm:
            clip_grad_norm_(self.distributions.parameters(), self.clip_norm)
        if self.clip_value:
            clip_grad_value_(self.distributions.parameters(), self.clip_value)

        self.optimizer.step()

        return loss

    # override "https://github.com/masa-su/pixyz/blob/v0.3.3/pixyz/models/model.py"
    def test(self, test_input_dict={}, include_missing=True, active_x_mask=None, **kwargs):
        self.distributions.eval()

        if include_missing:
            with torch.no_grad():
                loss = self.calc_dynamic_loss(test_input_dict, active_x_mask, **kwargs)
        else:
            with torch.no_grad():
                loss = self.test_loss_cls.eval(test_input_dict, **kwargs)

        return loss

    @abstractmethod
    def get_network_dicts(self):
        raise NotImplementedError()

    @abstractmethod
    def set_dist_dict(self):
        raise NotImplementedError()

    @abstractmethod
    def calc_dynamic_loss(self):
        raise NotImplementedError()    

    @abstractmethod
    def get_static_loss_cls(self):
        raise NotImplementedError()
        
    @abstractmethod
    def get_recon_dict(self, x_dict):
        raise NotImplementedError()