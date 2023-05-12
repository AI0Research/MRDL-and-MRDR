import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertModel, BertPreTrainedModel


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, freeze_bert=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        self.sim_fct = nn.CosineSimilarity(dim=-1)
        self.loss_fct_mse = nn.MSELoss()
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
        input_ids_pair=None,
        token_type_ids_pair=None,
        attention_mask_pair=None,
        labels=None
    ):
        if getattr(self, 'is_export', False):
            attention_mask = (input_ids != 0).type(input_ids.dtype)
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            pooled_output = outputs[1]
        else:
            outputs_query = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            outputs_title = self.bert(
                input_ids=input_ids_pair,
                attention_mask=attention_mask_pair,
                token_type_ids=token_type_ids_pair
            )
            query_embedding = outputs_query[1]
            title_embedding = outputs_title[1]

        logits = None
        loss = None

        if query_embedding is not None and title_embedding is not None and labels is not None:
            # simcse
            batch_emb = torch.cat([query_embedding, title_embedding], dim=0)  # (B*4, hidden_size)
            batch_size = batch_emb.size(0)
            idxs = torch.arange(0, batch_size).to(labels.device)
            y_true = idxs + 1 - idxs % 2 * 2
            sim_score = F.cosine_similarity(batch_emb.unsqueeze(1), batch_emb.unsqueeze(0), dim=2)
            sim_score = sim_score - torch.eye(batch_size).to(labels.device) * 1e12
            sim_score = sim_score / 0.05
            simcse_loss = self.loss_fct(sim_score, y_true)
            self._add_loss_item('simcse_loss', simcse_loss.item())

            # supervised
            sim_score = self.sim_fct(query_embedding, title_embedding)
            y_true = labels
            y_true[y_true <= 1] = 0
            y_true[y_true >= 2] = 1
            mse_loss = self.loss_fct_mse(sim_score, y_true)
            self._add_loss_item('mse_loss', mse_loss.item())

            loss = simcse_loss + mse_loss

            # HACK
            pooled_output = query_embedding
            outputs = outputs_query

        output = (logits, pooled_output) + outputs[2:]
        if getattr(self, 'is_export', False):
            return output[0]
        return ((loss,) + output) if loss is not None else output
