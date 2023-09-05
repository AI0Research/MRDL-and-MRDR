import torch


class FGSM:
    """
    Refer to the paper: Explaining and Harnessing Adversarial Examples

    Usage:
        adv_model = FGSM(model)
        outputs = model(**inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        adv_model.attack()
        outputs = model(**inputs)
        adv_loss = F.cross_entropy(outputs,labels)
        adv_loss.backward()
        adv_model.restore()
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

    Usage:
        Refer to FGSM
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

    Usage:
        adv_model = PGD(model)
        outputs = model(**inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        adv_k = 3
        for k in range(adv_k):
            adv_model.attack(is_first_attack=(k == 0))
            if k != adv_k - 1:
                model.zero_grad()
            else:
                adv_model.restore_grad()
            outputs = model(**inputs)
            adv_loss = F.cross_entropy(outputs, labels)
            adv_loss.backward()
        adv_model.restore()
    """

    def __init__(self, model, eps=1.0, alpha=0.3):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, emb_name="embedding", is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
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


class FreeAT:
    """
    Refer to the paper: Free Adversarial Training

    Usage:
        adv_model = FreeAT(model)
        outputs = model(**inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        adv_k = 5
        for k in range(adv_k):
            adv_model.attack(is_first_attack=(k == 0))
            if k != adv_k - 1:
                model.zero_grad()
            else:
                adv_model.restore_grad()
            outputs = model(**inputs)
            adv_loss = F.cross_entropy(outputs, labels)
            adv_loss.backward()
        adv_model.restore()
    """

    def __init__(self, model, eps=0.1):
        self.model = model
        self.eps = eps
        self.emb_backup = {}
        self.grad_backup = {}
        self.last_r_at = 0

    def attack(self, emb_name='embedding', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                param.data.add_(self.last_r_at)
                param.data = self.project(name, param.data)
                self.last_r_at = self.last_r_at + self.eps * param.grad.sign()

    def restore(self, emb_name='embedding'):
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
