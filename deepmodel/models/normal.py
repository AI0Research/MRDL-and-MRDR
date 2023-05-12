import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.char_bilstm import CharBiLSTM
from src.utils.math_utils import log_sum_exp
from src.utils.math_utils import argmax
from src.common.config import PAD_TAG, UNK_TAG


class CRF(nn.Module):
    def __init__(self,
                 num_tags,
                 label2idx,
                 idx2labels,
                 device = "cpu"):
        super().__init__()
        self.num_tags = num_tags
        self.idx2labels = idx2labels
        self.PAD_INDEX = label2idx[PAD_TAG]
 
        # CRF
        init_transition = torch.randn(self.num_tags, self.num_tags)
        init_transition[:, self.PAD_INDEX] = -10000.0
        init_transition[self.PAD_INDEX, :] = -10000.0

        self.transitions = nn.Parameter(init_transition)
        self.start_transitions = torch.nn.Parameter(torch.Tensor(self.num_tags))
        self.end_transitions = torch.nn.Parameter(torch.Tensor(self.num_tags))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.transitions)
        torch.nn.init.normal_(self.start_transitions)
        torch.nn.init.normal_(self.end_transitions)

    def forward(self, feats, tags, mask, _):
        gold_score = self._score_sentence(feats, tags, mask)
        forward_score = self._forward_alg(feats, mask)
        return torch.sum(forward_score - gold_score)

    def _forward_alg(self, feats, mask):
        """
        Parameters:
            feats: (batch_size, sequence_length, num_tags)
            mask: Show padding tags. 0 don't calculate score. (batch_size, sequence_length)
        Returns:
            scores: (batch_size)
        """
        batch_size, sequence_length, num_tags = feats.data.shape

        feats = feats.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()

        # Start transition score and first feats
        alpha = self.start_transitions.view(1, num_tags) + feats[0]

        for i in range(1, sequence_length):

            feats_score = feats[i].view(batch_size, 1, num_tags)             # (batch_size, 1, num_tags)
            transition_scores = self.transitions.view(1, num_tags, num_tags) # (1, num_tags, num_tags)
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)            # (batch_size, num_tags, 1)

            inner = broadcast_alpha + feats_score + transition_scores        # (batch_size, num_tags, num_tags)

            alpha = (log_sum_exp(inner, 1) * mask[i].view(batch_size, 1) +
                     alpha * (1 - mask[i]).view(batch_size, 1))

        # Add end transition score
        stops = alpha + self.end_transitions.view(1, num_tags)

        # Sum (log-sum-exp) over all possible tags
        return log_sum_exp(stops) # (batch_size,)

    def _score_sentence(self, feats, tags, mask):
        """
        Parameters:
            feats: (batch_size, sequence_length, num_tags)
            tags:  (batch_size, sequence_length)
            mask:  Show padding tags. 0 don't calculate score. (batch_size, sequence_length)
        Returns:
            scores: (batch_size)
        """

        batch_size, sequence_length, _ = feats.data.shape

        feats = feats.transpose(0, 1).contiguous()
        tags = tags.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()

        # Start transition score and first emission
        score = self.start_transitions.index_select(0, tags[0])

        for i in range(sequence_length - 1):
            current_tag, next_tag = tags[i], tags[i+1]

            # Transition score from current_tag to next_tag
            transition_score = self.transitions[current_tag.view(-1), next_tag.view(-1)]

            # feature score for next tag
            feats_score = feats[i].gather(1, current_tag.view(batch_size, 1)).squeeze(1)

            score = score + transition_score * mask[i + 1] + feats_score * mask[i]

        # Add end transition score
        last_tag_index = mask.sum(0).long() - 1
        last_tags = tags.gather(0, last_tag_index.view(1, batch_size)).squeeze(0)

        # Compute score of transitioning to STOP_TAG from each LAST_TAG
        last_transition_score = self.end_transitions.index_select(0, last_tags)

        last_inputs = feats[-1]                                          # (batch_size, num_tags)
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))  # (batch_size, 1)
        last_input_score = last_input_score.squeeze()                    # (batch_size,)

        score = score + last_transition_score + last_input_score * mask[-1]

        return score

    def _viterbi_tags(self, feats, mask):
        """
        Parameters:
            feats: (batch_size, sequence_length, num_tags)
            mask:  Show padding tags. 0 don't calculate score. (batch_size, sequence_length)
        Returns:
            tags: (batch_size)
        """
        batch_size, sequence_length, _ = feats.shape

        feats = feats.transpose(0, 1).contiguous()
        mask = mask.transpose(0, 1).contiguous()

        # Start transition and first emission
        score = self.start_transitions + feats[0]
        history = []

        for i in range(1, sequence_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_feats = feats[i].unsqueeze(1)

            next_score = broadcast_score + self.transitions + broadcast_feats
            next_score, indices = next_score.max(dim=1)

            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # Add end transition score
        score += self.end_transitions

        # Compute the best path for each sample
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            best_tags.reverse()
            best_tags_list.append(best_tags)

        return 