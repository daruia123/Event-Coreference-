import torch
from overrides import overrides

from allennlp.common.params import Params
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.nn import util


class SelfAttentiveSentenceExtractor(torch.nn.Module):

        def __init__(self,
                     input_dim: int) -> None:
            super().__init__()
            self._input_dim = input_dim
            self._global_attention = TimeDistributed(torch.nn.Linear(input_dim, 1))

        def get_input_dim(self) -> int:
            return self._input_dim

        def get_output_dim(self) -> int:
            return self._input_dim

        @overrides
        def forward(self,
                    sentence_tensor: torch.FloatTensor,
                    sentence_indices: torch.LongTensor,
                    sentence_mask: torch.LongTensor = None,
                    sentence_indices_mask: torch.LongTensor = None) -> torch.FloatTensor:
            # both of shape (batch_size, num_spans, 1)
            span_starts, span_ends = sentence_indices.split(1, dim=-1)

            # shape (batch_size, num_spans, 1)
            # These span widths are off by 1, because the span ends are `inclusive`.
            span_widths = span_ends - span_starts

            # We need to know the maximum span width so we can
            # generate indices to extract the spans from the sequence tensor.
            # These indices will then get masked below, such that if the length
            # of a given span is smaller than the max, the rest of the values
            # are masked.
            max_batch_span_width = int(span_widths.max().data) + 1

            # shape (batch_size, sequence_length, 1)
            global_attention_logits = self._global_attention(sentence_tensor)

            # Shape: (1, 1, max_batch_span_width)
            max_span_range_indices = util.get_range_vector(max_batch_span_width,
                                                           util.get_device_of(sentence_tensor)).view(1, 1, -1)
            # Shape: (batch_size, num_spans, max_batch_span_width)
            # This is a broadcasted comparison - for each span we are considering,
            # we are creating a range vector of size max_span_width, but masking values
            # which are greater than the actual length of the span.
            #
            # We're using <= here (and for the mask below) because the span ends are
            # inclusive, so we want to include indices which are equal to span_widths rather
            # than using it as a non-inclusive upper bound.
            span_mask = (max_span_range_indices <= span_widths).float()
            raw_span_indices = span_ends - max_span_range_indices
            # We also don't want to include span indices which are less than zero,
            # which happens because some spans near the beginning of the sequence
            # have an end index < max_batch_span_width, so we add this to the mask here.
            span_mask = span_mask * (raw_span_indices >= 0).float()
            span_indices = torch.nn.functional.relu(raw_span_indices.float()).long()

            # Shape: (batch_size * num_spans * max_batch_span_width)
            flat_span_indices = util.flatten_and_batch_shift_indices(span_indices, sentence_tensor.size(1))

            # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
            span_embeddings = util.batched_index_select(sentence_tensor, span_indices, flat_span_indices)

            # Shape: (batch_size, num_spans, max_batch_span_width)
            span_attention_logits = util.batched_index_select(global_attention_logits,
                                                              span_indices,
                                                              flat_span_indices).squeeze(-1)
            # Shape: (batch_size, num_spans, max_batch_span_width)
            span_attention_weights = util.last_dim_softmax(span_attention_logits, span_mask)

            # Do a weighted sum of the embedded spans with
            # respect to the normalised attention distributions.
            # Shape: (batch_size, num_spans, embedding_dim)
            attended_text_embeddings = util.weighted_sum(span_embeddings, span_attention_weights)

            if sentence_indices_mask is not None:
                # Above we were masking the widths of spans with respect to the max
                # span width in the batch. Here we are masking the spans which were
                # originally passed in as padding.
                return attended_text_embeddings * sentence_indices_mask.unsqueeze(-1).float()

            return attended_text_embeddings

        @classmethod
        def from_params(cls, params: Params) -> "SelfAttentiveSentenceExtractor":
            input_dim = params.pop_int("input_dim")
            params.assert_empty(cls.__name__)
            return SelfAttentiveSentenceExtractor(input_dim=input_dim)
