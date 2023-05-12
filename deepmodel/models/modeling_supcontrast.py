import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertModel, BertPreTrainedModel


class SupConLossPLMS(nn.Module):
    """Supervised Contrastive Learning for Pre-trained Language Model Fine-tuning: https://arxiv.org/abs/2011.01403
    """

    def __init__(self, temperature=0.3):
        super(SupConLossPLMS, self).__init__()
        self.temperature = temperature

    def forward(self, batch_emb, labels=None):
        labels = labels.view(-1, 1)
        batch_size = batch_emb.shape[0]
        mask = torch.eq(labels, labels.T).float()
        norm_emb = F.normalize(batch_emb, dim=1, p=2)
        # compute logits
        # dot_contrast = F.cosine_similarity(norm_emb[:,:,None], norm_emb.t()[None,:,:])
        dot_contrast = torch.div(torch.matmul(norm_emb, norm_emb.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(dot_contrast, dim=1, keepdim=True)  # _返回索引
        logits = dot_contrast - logits_max.detach()
        # 索引应该保证设备相同
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(batch_emb.device), 0)
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mask_sum = mask.sum(1)
        # 防止出现NAN
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = -(mask * log_prob).sum(1) / mask_sum
        return mean_log_prob_pos.mean()


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, freeze_bert=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct_ce = nn.CrossEntropyLoss()
        self.loss_fct_scl = SupConLossPLMS(temperature=0.3)
        self.init_weights()

    def _add_loss_item(self, key, value):
        if not hasattr(self, 'loss_dict'):
            self.loss_dict = {}
        if key not in self.loss_dict:
            self.loss_dict[key] = [value, 1]
        else:
            last_value_avg, last_cnt = self.loss_dict[key]
            self.loss_dict[key] = [(last_value_avg*last_cnt+value)/(last_cnt+1), last_cnt+1]

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
    ):
        if getattr(self, 'is_export', False):
            attention_mask = (input_ids != 0).type(input_ids.dtype)
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)

        loss = None

        if labels is not None:
            ce_loss = self.loss_fct_ce(logits, labels)
            self._add_loss_item('ce_loss', ce_loss.item())
            scl_loss = self.loss_fct_scl(batch_emb=pooled_output, labels=labels)
            self._add_loss_item('scl_loss', scl_loss.item())
            loss = 0.1 * ce_loss + 0.9 * scl_loss
            self._add_loss_item('loss', loss.item())

        output = (logits, pooled_output) + outputs[2:]
        if getattr(self, 'is_export', False):
            return output[0]
        return ((loss,) + output) if loss is not None else output
