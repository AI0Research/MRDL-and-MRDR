import math
from typing import Callable, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
from torch.optim import Optimizer


class ChildTuningAdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        reserve_p=1.0,
        mode=None
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

        self.gradient_mask = None
        self.reserve_p = reserve_p
        self.mode = mode

    def set_gradient_mask(self, gradient_mask):
        self.gradient_mask = gradient_mask

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.
        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # =================== HACK BEGIN =======================
                if self.mode is not None:
                    if self.mode == 'ChildTuning-D':
                        if p in self.gradient_mask:
                            grad *= self.gradient_mask[p]
                    else:
                        # ChildTuning-F
                        grad_mask = Bernoulli(grad.new_full(size=grad.size(), fill_value=self.reserve_p))
                        grad *= grad_mask.sample() / self.reserve_p
                # =================== HACK END =======================

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss


class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target, reduction='mean'):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        if reduction == 'mean':
            loss = (-weight * log_prob).sum(dim=-1).mean()
        elif reduction == 'none':
            loss = (-weight * log_prob).sum(dim=-1)
        return loss


class NTXentLoss(torch.nn.Module):
    """
        SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
        Refer: https://github.com/sthalles/SimCLR/blob/master/loss/nt_xent.py
    """

    def __init__(self, batch_size, temperature=0.5, use_cosine_similarity=True):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        #self.device = torch.device("cuda:0") if next(self.parameters()).is_cuda else "cpu"
        #self.device = torch.device("cuda:0")
        #self.device = torch.device("cpu")
        self.mask_samples_from_same_repr = None
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        # zis: [bs, logit_dim], zjs: [bs, logit_dim]
        # Lazy init mask
        if self.mask_samples_from_same_repr is None:
            self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool).to(zis.device)
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(logits.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]


class FocalCeLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1, reduction='mean'):
        """https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py"""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = 1e-6
        self.reduction = reduction

    def forward(self, input, target):
        input_soft = F.softmax(input) + self.epsilon
        target_one_hot = torch.zeros_like(input)
        target_one_hot[torch.arange(target.size(0)), target] = 1

        # compute the actual focal loss
        weight = torch.pow(-input_soft + 1.0, self.gamma)
        focal = -self.alpha * weight * torch.log(input_soft)
        loss_tmp = torch.sum(target_one_hot * focal, dim=1)

        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError
        return loss

        prob = F.sigmoid(input)
        pt = (1 - prob) * target + prob * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * (pt ** self.gamma)
        bce_fct = nn.BCEWithLogitsLoss(reduce=None)
        loss = bce_fct(input, target) * focal_weight

        if self.size_average:
            return torch.mean(loss)
        else:
            return torch.sum(loss)


class FocalLoss(nn.Module):
    """https://github.com/wutong16/DistributionBalancedLoss/blob/master/mllt/models/losses/focal_loss.py#L9"""

    def __init__(self, gamma=0, alpha=1, is_logits=False, size_average=True):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = 1e-9
        self.is_logits = is_logits
        self.size_average = size_average

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.size_average:
            return torch.mean(F_loss)
        else:
            return torch.sum(F_loss)


class Ghmc(nn.Module):
    """https://github.com/libuyu/mmdetection/blob/be06992564cc6b995b1ae86a258568e9d7b7a599/mmdet/models/losses/ghm_loss.py"""

    def __init__(
            self,
            bins=10,
            momentum=0,
            use_sigmoid=True,
            loss_weight=1.0):
        super().__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = [float(x) / bins for x in range(bins+1)]
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight

    @classmethod
    def _expand_binary_labels(cls, labels, label_weights, label_channels):
        bin_labels = labels.new_full((labels.size(0), label_channels), 0)
        inds = torch.nonzero(labels >= 1).squeeze()
        if inds.numel() > 0:
            bin_labels[inds, labels[inds] - 1] = 1
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)
        return bin_labels, bin_label_weights

    def forward(self, pred, target, label_weight=None, *args, **kwargs):
        """ Args:
        pred [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            Binary class target for each sample.
        label_weight [batch_num, class_num]:
            the value is 1 if the sample is valid and 0 if ignored.
        """
        if not self.use_sigmoid:
            raise NotImplementedError
        if label_weight is None:
            label_weight = torch.ones_like(pred)
        # the target should be binary class label
        if pred.dim() != target.dim():
            target, label_weight = self._expand_binary_labels(target, label_weight, pred.size(-1))
        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        g = torch.abs(pred.sigmoid().detach() - target)

        valid = label_weight > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(
            pred, target, weights, reduction='sum') / tot
        return loss * self.loss_weight


#=============================FM=============================#


class CrossNet(nn.Module):
    """The Cross Network part of Deep&Cross Network model,
    which leans both low and high degree cross feature.
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Arguments
        - **in_features** : Positive integer, dimensionality of input features.
        - **input_feature_num**: Positive integer, shape(Input tensor)[-1]
        - **layer_num**: Positive integer, the cross layer number
        - **parameterization**: string, ``"vector"``  or ``"matrix"`` ,  way to parameterize the cross network.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix
        - **seed**: A Python integer to use as random seed.
      References
        - [Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12.](https://arxiv.org/abs/1708.05123)
        - [Wang R, Shivanna R, Cheng D Z, et al. DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems[J]. 2020.](https://arxiv.org/abs/2008.13535)
        - https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/layers/interaction.py#L406
    """

    def __init__(self, in_features, layer_num=2, parameterization='vector', seed=1024, device='cpu'):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        self.parameterization = parameterization
        if self.parameterization == 'vector':
            # weight in DCN.  (in_features, 1)
            self.kernels = torch.nn.ParameterList(
                [nn.Parameter(nn.init.xavier_normal_(torch.empty(in_features, 1))) for i in range(self.layer_num)])
        elif self.parameterization == 'matrix':
            # weight matrix in DCN-M.  (in_features, in_features)
            self.kernels = torch.nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(
                torch.empty(in_features, in_features))) for i in range(self.layer_num)])
        else:  # error
            raise NotImplementedError("parameterization should be 'vector' or 'matrix'")

        self.bias = torch.nn.ParameterList(
            [nn.Parameter(nn.init.zeros_(torch.empty(in_features, 1))) for i in range(self.layer_num)])

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
            if self.parameterization == 'vector':
                xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
                dot_ = torch.matmul(x_0, xl_w)
                x_l = dot_ + self.bias[i]
            elif self.parameterization == 'matrix':
                dot_ = torch.matmul(self.kernels[i], x_l)  # W * xi  (bs, in_features, 1)
                dot_ = dot_ + self.bias[i]  # W * xi + b
                dot_ = x_0 * dot_  # x0 · (W * xi + b)  Hadamard-product
            else:  # error
                raise NotImplementedError("parameterization should be 'vector' or 'matrix'")
            x_l = dot_ + x_l
        x_l = torch.squeeze(x_l, dim=2)
        return x_l


class FMSelf(nn.Module):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
        - https://github.com/shenweichen/DeepCTR-Torch/blob/6eec1edaf0e1cc206998a57a348539d287d7c351/deepctr_torch/layers/interaction.py#L12
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        fm_input = inputs

        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)  # [bs, fs, es]->[bs, 1, es]^2
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)  # [bs, fs, es]->[bs, 1, es]
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)

        return cross_term


class FM(nn.Module):
    def __init__(self, feat_size=None, emb_dim=None):
        """Refer: https://www.kaggle.com/gennadylaptev/factorization-machine-implemented-in-pytorch"""
        super().__init__()
        # Initially we fill V with random values sampled from Gaussian distribution
        # NB: use nn.Parameter to compute gradients
        self.V = nn.Parameter(torch.randn(emb_dim, feat_size), requires_grad=True)
        #self.lin = nn.Linear(emb_dim, 1)

    def forward(self, x):
        square_of_sum = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True)  # S_1^2 [bs, 1, es]
        sum_of_square = torch.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True)  # S_2 [bs, 1, es]

        cross_term = 0.5 * (square_of_sum - sum_of_square)  # [bs, 1, es]
        out = cross_term
        #out_lin = self.lin(x)
        #out = cross_term + out_lin
        return out


class FMBitwise(nn.Module):
    def __init__(self, feat_size=None, v_emb_dim=128):
        """Refer: https://www.kaggle.com/gennadylaptev/factorization-machine-implemented-in-pytorch"""
        super().__init__()
        # Initially we fill V with random values sampled from Gaussian distribution
        # NB: use nn.Parameter to compute gradients
        self.V = nn.Parameter(torch.randn(feat_size, v_emb_dim), requires_grad=True)
        self.fm_dim = feat_size * feat_size

    def forward(self, x):
        # [bs, es]
        xij = torch.bmm(x.unsqueeze(dim=2), x.unsqueeze(dim=1))  # [bs, es, 1][bs, 1, es]->[bs, es, es]
        vij = torch.matmul(self.V, torch.transpose(self.V, 0, 1))  # [fs, es][es, fs]->[fs, fs]
        vij = vij.unsqueeze(dim=0)
        fij = vij * xij
        fij = fij.view(fij.size(0), -1)
        return fij


#=============================GIN=============================#

class GINConv(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, A, X):
        """
        Params
        ------
        A [batch x nodes x nodes]: adjacency matrix
        X [batch x nodes x features]: node features matrix

        Returns
        -------
        X' [batch x nodes x features]: updated node features matrix
        """
        #X = self.linear(X + A @ X + A**2 @ X)
        X = self.linear(X + A @ X)
        X = torch.nn.functional.relu(X)

        return X


class GINConv(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, A, X):
        """
        Params
        ------
        A [batch x nodes x nodes]: adjacency matrix
        X [batch x nodes x features]: node features matrix

        Returns
        -------
        X' [batch x nodes x features]: updated node features matrix
        """
        X = self.linear(X + A @ X)
        X = torch.nn.functional.relu(X)

        return X


class GIN(torch.nn.Module):
    def __init__(self, nfeat, hidden_dim, n_layers=1):
        super().__init__()
        output_dim = hidden_dim
        self.in_proj = torch.nn.Linear(nfeat, hidden_dim)

        self.convs = torch.nn.ModuleList()

        for _ in range(n_layers):
            self.convs.append(GINConv(hidden_dim))

        # In order to perform graph classification, each hidden state
        # [batch x nodes x hidden_dim] is concatenated, resulting in
        # [batch x nodes x hiddem_dim*(1+n_layers)], then aggregated
        # along nodes dimension, without keeping that dimension:
        # [batch x hiddem_dim*(1+n_layers)].
        self.out_proj = torch.nn.Linear(hidden_dim*(1+n_layers), output_dim)

    def forward(self, X, A):
        X = self.in_proj(X)

        hidden_states = [X]

        for layer in self.convs:
            X = layer(A, X)
            hidden_states.append(X)

        X = torch.cat(hidden_states, dim=1)

        X = self.out_proj(X)
        # raise

        return X

#=============================GCN=============================#


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, proj=True, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if proj:
            self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        else:
            self.register_parameter('weight', None)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        if self.weight is not None:
            support = torch.mm(input, self.weight)
        else:
            support = input
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout, proj=True, bias=True, act=True):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, proj=proj, bias=bias)
        #self.gc1 = GraphConvolution(nfeat, nhid)
        #self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.act = act

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        if self.act:
            x = F.relu(x)
        return x
        #x = F.dropout(x, self.dropout, training=self.training)
        #x = self.gc2(x, adj)
        # return F.log_softmax(x, dim=1)


#=============================GAT=============================#

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.6,
                 alpha=0.2, nheads=8):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x
        # return F.log_softmax(x, dim=1)


#=============================Encoder=============================#

class RnnEncoder(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            embed_size, hidden_size // 2, num_layers,
            bidirectional=True, batch_first=True, dropout=dropout
        )
        #self.maxpool = nn.MaxPool1d(pad_size)

    def forward(self, inputs_embeds=None):
        lstm_out, _ = self.lstm(inputs_embeds)
        #hidden_embeds = torch.cat((inputs_embeds, lstm_out), dim=2)
        hidden_embeds = inputs_embeds + lstm_out
        hidden_embeds = F.relu(hidden_embeds)
        hidden_embeds = torch.mean(hidden_embeds, dim=1)
        return hidden_embeds


class CnnEncoder(nn.Module):
    def __init__(self, hidden_size, num_filters, filter_sizes, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, hidden_size)) for k in filter_sizes]
        )

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, inputs_embeds=None):
        seq_emb = inputs_embeds.unsqueeze(dim=1)
        cnn_output = torch.cat([self.conv_and_pool(seq_emb, conv) for conv in self.convs], 1)
        cnn_output = self.dropout(cnn_output)
        return cnn_output


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z, return_beta=False):
        # [bs, M, 64]->[bs, M, 128]->[bs, M, 1]->[M, 1]
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        ret = (beta * z).sum(1)                       # (N, D * K)
        if return_beta:
            return ret, beta
        else:
            return ret


if __name__ == "__main__":
    pass
