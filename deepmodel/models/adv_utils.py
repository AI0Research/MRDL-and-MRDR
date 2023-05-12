import torch


class FGSM:
    """
    Refer to the paper: Explaining and Harnessing Adversarial Examples
    """

    def __init__(self, model, eps=1.0):
        self.model = model
        self.eps = eps
        self.backup = {}

    def attack(self, emb_name='embedding'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                r_at = self.eps * param.grad.sign()
                param.data.add_(r_at)

    def restore(self, emb_name='embedding'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class FGM():
    """
    Refer to the paper: Adversarial training methods for semi-supervised text classification
    """

    def __init__(self, model, eps=1.0):
        self.model = model
        self.eps = eps
        self.backup = {}

    def attack(self, emd_name="embedding"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emd_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.eps * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emd_name="embedding"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emd_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD():
    """
    Refer to the paper: Towards Deep Learning Models Resistant to Adversarial Attacks
    """

    def __init__(self, model, eps=1.0, alpha=0.3):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, emb_name="embedding", backup=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if backup:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data)

    def restore(self, emb_name="embedding"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > self.eps:
            r = self.eps * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


AdvModel = {
    'fgsm': FGSM,
    'fgm': FGM,
    'pgd': PGD
}
