from typing import List, Optional

import torch
import torch.nn as nn
import numpy as np

"""
CRF module adapted from https://pytorch-crf.readthedocs.io/en/stable/_modules/torchcrf.html#CRF
"""

def log_sum_exp(x):
    """calculate log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    """
    max_score = x.max(-1)[0]
    return max_score + (x - max_score.unsqueeze(-1)).exp().sum(-1).log()


IMPOSSIBLE = -1e4

class CRF(nn.Module):
    """Conditional random field.

    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.

    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.

    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.


    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.

    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self, num_tags: int, seq: int, transmat: float = None, batch_first: bool = True, 
                 learn_ind: bool = False, stop_grad=False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.learn_ind = learn_ind
        self.seq = seq
        self.stop_grad = stop_grad
        
        # transition factor, Tij mean transition from j to i
        if self.learn_ind and not stop_grad:
            self.transitions = nn.Parameter(torch.randn(self.num_tags, self.num_tags, self.seq-1), requires_grad=True)
            self.start_transitions = nn.Parameter(torch.empty(num_tags))
            self.end_transitions = nn.Parameter(torch.empty(num_tags))
            self.reset_parameters()
        elif stop_grad:
            self.transitions = nn.Parameter(torch.Tensor(transmat), requires_grad=True)
            self.start_transitions = nn.Parameter(torch.empty(num_tags))
            self.end_transitions = nn.Parameter(torch.empty(num_tags))    
            self.reset_startend()                  
        else:
            self.transitions = torch.Tensor(transmat).float().cuda()
            self.start_transitions = torch.Tensor(np.zeros(num_tags)).float().cuda()
            self.end_transitions = torch.Tensor(np.zeros(num_tags)).float().cuda()


    def reset_parameters(self) -> None:
        """Initialize the transition parameters.

        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        
    def reset_startend(self) -> None:
        """Initialize the transition parameters.

        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def loss(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: Optional[torch.ByteTensor] = None,
            reduction: str = 'mean',
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.

        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        emissions, tags = self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = denominator - numerator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'
        return llh.sum() / mask.float().sum()


    def forward(self, emissions: torch.Tensor,
                tags: Optional[torch.LongTensor] = None,
                mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.

        Returns:
            List of list containing the best tag sequence for each batch.
        """
        emissions, _ = self._validate(emissions, tags=tags, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)


    def _validate(self,
            emissions: torch.Tensor,
            tags: Optional[torch.LongTensor] = None,
            mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() !=5:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
            
        B, L, C, H, W = emissions.shape
        emissions = emissions.permute(0,3,4,1,2).contiguous()
        emissions = emissions.view(B*H*W, L, C)
        
        try:       
            tags = tags.permute(0,2,3,1).contiguous()
            tags = tags.view(B*H*W, L)
            
            emissions = emissions[tags[:,0]<C]
            tags = tags[tags[:,0]<C]
          
            if emissions.size(2) != self.num_tags:
                raise ValueError(
                    f'expected last dimension of emissions is {self.num_tags}, '
                    f'got {emissions.size(2)}')
    
            if tags is not None:
                if emissions.shape[:2] != tags.shape:
                    raise ValueError(
                        'the first two dimensions of emissions and tags must match, '
                        f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')
        
            if mask is not None:
                if emissions.shape[:2] != mask.shape:
                    raise ValueError(
                        'the first two dimensions of emissions and mask must match, '
                        f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
                no_empty_seq = not self.batch_first and mask[0].all()
                no_empty_seq_bf = self.batch_first and mask[:, 0].all()
                if not no_empty_seq and not no_empty_seq_bf:
                    raise ValueError('mask of the first timestep must all be on')
                
            return emissions, tags
        except:
            if mask is not None:
                if emissions.shape[:2] != mask.shape:
                    raise ValueError(
                        'the first two dimensions of emissions and mask must match, '
                        f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
                no_empty_seq = not self.batch_first and mask[0].all()
                no_empty_seq_bf = self.batch_first and mask[:, 0].all()
                if not no_empty_seq and not no_empty_seq_bf:
                    raise ValueError('mask of the first timestep must all be on')
                
            return emissions, 0             

    def _compute_score(
            self, emissions: torch.Tensor, tags: torch.LongTensor,
            mask: torch.ByteTensor) -> torch.Tensor:

        seq_length, batch_size = tags.shape
        mask = mask.float()

        # Start transition score and first emission
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            score += self.transitions[tags[i - 1], tags[i], i-1] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        seq_ends = mask.long().sum(dim=0) - 1

        last_tags = tags[seq_ends, torch.arange(batch_size)]

        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
            self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:

        seq_length = emissions.size(0)

        # Start transition score and first emission
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            next_score = broadcast_score + self.transitions[:,:,i-1] + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # End transition score
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor) -> List[List[int]]:

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        score = self.start_transitions + emissions[0]
        history = []

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor
            next_score = broadcast_score + self.transitions[:,:,i-1] + broadcast_emission

            # Find the maximum score over all possible current tag
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # End transition score
        score += self.end_transitions

        # Now, compute the best path for each sample

        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return np.asarray(best_tags_list)