import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel


class BertModelBase(BertPreTrainedModel):

    def __init__(self, config, add_pooling_layer=True, freeze_bert=False):
        super().__init__(config)
        self.bert = BertModel(
            config=config,
            add_pooling_layer=add_pooling_layer
        )
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.init_weights()

    def add_loss_item(self, key, value):
        self.loss_dict = getattr(self, 'loss_dict', {})
        if key not in self.loss_dict:
            self.loss_dict[key] = [value, 1]
        else:
            last_value_avg, last_cnt = self.loss_dict[key]
            self.loss_dict[key] = [(last_value_avg * last_cnt + value) / (last_cnt + 1), last_cnt + 1]

    def forward(
        self,
        *args,
        **kwargs
    ):
        raise NotImplementedError


class BertForSequenceClassification(BertModelBase):

    def __init__(self, config, freeze_bert=False):
        super().__init__(
            config=config,
            freeze_bert=freeze_bert
        )
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)

        loss = None

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            self.add_loss_item('loss', loss.item())

        outputs = (logits, pooled_output) + outputs[2:]
        if getattr(self, 'is_export', False):
            return outputs[0]
        return ((loss,) + outputs) if loss is not None else outputs


class BertForTokenClassification(BertModelBase):

    def __init__(self, config, freeze_bert=False):
        super().__init__(
            config=config,
            add_pooling_layer=False,
            freeze_bert=freeze_bert
        )
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None

        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            self.add_loss_item("loss", loss.item())

        outputs = (logits, sequence_output) + outputs[2:]
        if getattr(self, 'is_export', False):
            return outputs[0]
        return ((loss,) + outputs) if loss is not None else outputs

    def decode(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        logits = torch.argmax(logits, dim=-1)
        return logits
