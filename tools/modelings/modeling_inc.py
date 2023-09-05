import copy

import torch.nn.functional as F
from torch import nn
from transformers import BertModel, BertPreTrainedModel


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, freeze_bert=False, *args, **kwargs):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        self.init_weights()

    def _add_loss_item(self, key, value):
        if not hasattr(self, 'loss_dict'):
            self.loss_dict = {}
        if key not in self.loss_dict:
            self.loss_dict[key] = [value, 1]
        else:
            last_value_avg, last_cnt = self.loss_dict[key]
            self.loss_dict[key] = [(last_value_avg*last_cnt+value)/(last_cnt+1), last_cnt+1]

    def add_teacher(self):
        self.teacher_bert = copy.deepcopy(self.bert)
        self.teacher_classifier = copy.deepcopy(self.classifier)
        for param in self.teacher_bert.parameters():
            param.requires_grad = False
        for param in self.teacher_classifier.parameters():
            param.requires_grad = False

    def del_teacher(self):
        self.teacher_bert = None
        self.teacher_classifier = None

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
            if self.teacher_bert is not None:
                teacher_outputs = self.teacher_bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                teacher_pooled_output = teacher_outputs[1]
                teacher_logits = self.teacher_classifier(teacher_pooled_output)
                teacher_pooled_output = teacher_pooled_output.detach()
                teacher_logits = teacher_logits.detach()

        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)

        loss = None

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            self._add_loss_item('loss', loss.item())
            if self.teacher_bert is not None:
                # feature distillation
                # distill_loss = F.mse_loss(student_pooled_output, teacher_pooled_output)
                # logits distillation
                distill_loss = F.kl_div(F.log_softmax(logits, dim=1), F.softmax(teacher_logits, dim=1))
                # distill_loss = F.mse_loss(logits, teacher_logits)
                self._add_loss_item('distill_loss', distill_loss.item())
                loss = loss + distill_loss

        output = (logits, pooled_output) + outputs[2:]
        if getattr(self, 'is_export', False):
            return output[0]
        return ((loss,) + output) if loss is not None else output
