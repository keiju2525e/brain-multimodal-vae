import os
import torch
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_dl, test_dl=None, eval=False):
        self.model = model
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.eval = eval

    @property
    def device(self):
        return self.model.device

    @property
    def subject_list(self):
        return self.train_dl.dataset.subject_list
    
    @property
    def include_missing(self):
        return self.train_dl.dataset.include_missing

    def train(self, n_epochs):
        for epoch in range(1, n_epochs + 1):
            train_loss = self._train_one_epoch()
            print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))

            if self.eval and (self.test_dl is not None):
                test_loss = self._test_one_epoch()
                print('Epoch: {} Test loss: {:.4f}'.format(epoch, test_loss))

    def save(self, params):
        save_dir = os.path.join(
            params["ckpt_dir"],
            f"{params['model_name']}",
            f"include_missing_{params['include_missing']}",
            f"train_group_{params['train_group']}"
        )
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(
            save_dir,
            f"{''.join(params['subject_list'])}_{params['n_train_repetitions']}_{params['n_shared_labels']}_{params['n_unique_labels']}_{params['select_seed']}.pt"
        )
        self.model.save(save_path)

    def _train_one_epoch(self):
        train_loss = 0
        total_samples = 0

        for brain_batch_dict, label_batch_dict, _ in tqdm(self.train_dl):
            for s in self.subject_list:
                brain_batch_dict[f"{s}"] = brain_batch_dict[f"{s}"].to(torch.float32).to(self.device)
                label_batch_dict[f"{s}"] = label_batch_dict[f"{s}"].to(self.device)

            active_x_mask_batch = torch.stack([(label_batch_dict[f"{s}"] != -1).int() for s in self.subject_list], dim=1)
            loss = self.model.train(brain_batch_dict, self.include_missing, active_x_mask_batch)

            n_batch_samples = next(iter(brain_batch_dict.values())).shape[0]
            train_loss += loss * n_batch_samples
            total_samples += n_batch_samples

        train_loss = train_loss / total_samples

        return train_loss

    def _test_one_epoch(self):
        test_loss = 0
        total_samples = 0

        with torch.no_grad():
            for brain_batch_dict, label_batch_dict, _ in tqdm(self.test_dl):
                for s in self.subject_list:
                    brain_batch_dict[f"{s}"] = brain_batch_dict[f"{s}"].to(torch.float32).to(self.device)
                    label_batch_dict[f"{s}"] = label_batch_dict[f"{s}"].to(self.device)

                active_x_mask_batch = torch.stack([(label_batch_dict[f"{s}"] != -1).int() for s in self.subject_list], dim=1)
                loss = self.model.test(brain_batch_dict, self.include_missing, active_x_mask_batch)

                n_batch_samples = next(iter(brain_batch_dict.values())).shape[0]
                test_loss += loss * n_batch_samples
                total_samples += n_batch_samples

        test_loss = test_loss / total_samples

        return test_loss