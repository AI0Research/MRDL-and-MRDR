import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothLoss(nn.Module):

    def __init__(self, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        if self.reduction == 'mean':
            loss = (-weight * log_prob).sum(dim=-1).mean()
        elif self.reduction == 'none':
            loss = (-weight * log_prob).sum(dim=-1)
        else:
            raise NotImplementedError
        return loss


class FocalLoss(nn.Module):
    '''
    Refer to the paper: Focal Loss for Dense Object Detection
    '''

    def __init__(self, gamma=2, alpha=1, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = 1e-6
        self.reduction = reduction

    def forward(self, input, target):
        input_soft = F.softmax(input, dim=-1) + self.epsilon
        target_one_hot = torch.zeros_like(input)
        target_one_hot[torch.arange(target.size(0)), target] = 1

        weight = torch.pow(-input_soft + 1.0, self.gamma)
        focal = -self.alpha * weight * torch.log(input_soft)
        loss = torch.sum(target_one_hot * focal, dim=1)

        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            raise NotImplementedError
        return loss


class DiceLoss(nn.Module):
    """
    Refer to the paper: Dice Loss for Data-imbalanced NLP Tasks
    """

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        prob = torch.softmax(input, dim=1)
        prob = torch.gather(prob, dim=1, index=target.unsqueeze(1))
        loss = (1 - ((1 - prob) * prob) / ((1 - prob) * prob + 1)).mean()
        return loss


class LogitAdjustmentLoss(nn.Module):
    '''
    Refer to the paper: Long-Tail Learning via Logit Adjustment

    Args:
        class_samples: [100, 200, 300, 300, 100]
    '''

    def __init__(self, class_samples, tau=1.0):
        super().__init__()
        total_samples = sum(class_samples)
        label_prob = [class_i_samples / total_samples for class_i_samples in class_samples]
        label_prob = torch.Tensor(label_prob).float()
        scaled_class_weights = tau * torch.log(label_prob + 1e-12)
        scaled_class_weights = scaled_class_weights.view(1, -1)
        self.scaled_class_weights = scaled_class_weights.float().cuda()

    def forward(self, input, target):
        input = input + self.scaled_class_weights
        return F.cross_entropy(input, target)


class CircleLoss(nn.Module):
    '''
    Refer to the paper: Circle Loss: A Unified Perspective of Pair Similarity Optimization
    '''

    def __init__(self, scale=32, margin=0.25, similarity='cos'):
        super(CircleLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.similarity = similarity

    def forward(self, feats, labels):
        m = labels.size(0)
        mask = labels.expand(m, m).t().eq(labels.expand(m, m)).float()
        pos_mask = mask.triu(diagonal=1)
        neg_mask = (mask - 1).abs_().triu(diagonal=1)
        if self.similarity == 'dot':
            sim_mat = torch.matmul(feats, torch.t(feats))
        elif self.similarity == 'cos':
            feats = F.normalize(feats)
            sim_mat = feats.mm(feats.t())
        else:
            raise NotImplementedError

        pos_pair_ = sim_mat[pos_mask == 1]
        neg_pair_ = sim_mat[neg_mask == 1]

        alpha_p = torch.relu(-pos_pair_ + 1 + self.margin)
        alpha_n = torch.relu(neg_pair_ + self.margin)
        margin_p = 1 - self.margin
        margin_n = self.margin
        loss_p = torch.sum(torch.exp(-self.scale * alpha_p * (pos_pair_ - margin_p)))
        loss_n = torch.sum(torch.exp(self.scale * alpha_n * (neg_pair_ - margin_n)))
        loss = torch.log(1 + loss_p * loss_n)
        return loss


class SupConLoss(nn.Module):
    '''
    Refer to the paper: Supervised Contrastive Learning for Pre-trained Language Model Fine-tuning
    '''

    def __init__(self, temperature=0.3):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, batch_emb, labels=None):
        labels = labels.view(-1, 1)
        batch_size = batch_emb.shape[0]
        mask = torch.eq(labels, labels.T).float()
        norm_emb = F.normalize(batch_emb, dim=1, p=2)
        dot_contrast = torch.div(torch.matmul(norm_emb, norm_emb.T), self.temperature)
        logits_max, _ = torch.max(dot_contrast, dim=1, keepdim=True)
        logits = dot_contrast - logits_max.detach()
        logits_idxs = torch.arange(batch_size).view(-1, 1).to(batch_emb.device)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, logits_idxs, 0)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = -(mask * log_prob).sum(1) / mask_sum
        return mean_log_prob_pos.mean()


class RDropLoss(nn.Module):
    '''
    Refer to the paper: R-Drop: Regularized Dropout for Neural Networks
    '''

    def __init__(self, alpha=2):
        super().__init__()
        self.alpha = alpha

    def forward(self, logits_p, logits_q):
        p_loss = F.kl_div(F.log_softmax(logits_p, dim=-1), F.softmax(logits_q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(logits_q, dim=-1), F.softmax(logits_p, dim=-1), reduction='none')
        p_loss = p_loss.mean()
        q_loss = q_loss.mean()
        loss = (p_loss + q_loss) / 2 * self.alpha
        return loss


class EWLossWeighting(nn.Module):

    def __init__(self, loss_num):
        super().__init__()
        self.loss_num = loss_num

    def forward(self, *inputs):
        assert len(inputs) > 0
        return (1 / self.loss_num) * sum(inputs)


class RandLossWeighting(nn.Module):

    def __init__(self, loss_num):
        super().__init__()
        self.loss_num = loss_num

    def forward(self, *inputs):
        assert len(inputs) > 0
        device = inputs[0].device
        losses = torch.stack(inputs)
        weight = F.softmax(torch.randn(self.loss_num), dim=-1).to(device)
        return torch.mul(losses, weight).sum()


class ScaleLossWeighting(nn.Module):

    def __init__(self, loss_num):
        super().__init__()
        self.loss_num = loss_num

    def forward(self, *inputs):
        assert len(inputs) > 0
        loss = inputs[0]
        for idx in range(1, len(inputs)):
            loss += inputs[idx] / (inputs[idx] / inputs[0]).detach()
        return loss


class UWLossWeighting(nn.Module):
    '''
    Refer to the paper: Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
    '''

    def __init__(self, loss_num):
        super().__init__()
        self.loss_num = loss_num
        self.log_sigma = nn.Parameter(torch.tensor([0.0] * self.loss_num))

    def forward(self, *inputs):
        assert len(inputs) > 0
        losses = torch.stack(inputs)
        return (losses / (2 * self.log_sigma.exp()) + self.log_sigma / 2).sum()
