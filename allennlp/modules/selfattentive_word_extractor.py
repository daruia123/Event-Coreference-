import torch
from overrides import overrides

from allennlp.common.params import Params
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.nn import util


class SelfAttentiveWordExtractor(torch.nn.Module):

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
                    text_tensor: torch.FloatTensor,
                    contextualized_embedding: torch.FloatTensor,
                    word_indices: torch.LongTensor,
                    word_mask: torch.LongTensor = None,
                    word_indices_mask: torch.LongTensor = None) -> torch.FloatTensor:
            # both of shape (batch_size, num_spans, 1)
            word = word_indices.unsqueeze(-1)

            global_attention_logits = self._global_attention(text_tensor)

            word_indices = torch.nn.functional.relu(word_indices.float()).long()

            # Shape: (batch_size * num_spans * max_batch_span_width)
            flat_word_indices = util.flatten_and_batch_shift_indices(word_indices, text_tensor.size(1))

            # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
            span_embeddings = util.batched_index_select(text_tensor, span_indices, flat_span_indices)

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

            if word_indices_mask is not None:
                # Above we were masking the widths of spans with respect to the max
                # span width in the batch. Here we are masking the spans which were
                # originally passed in as padding.
                return attended_text_embeddings * word_indices_mask.unsqueeze(-1).float()

            return attended_text_embeddings

        @classmethod
        def from_params(cls, params: Params) -> "SelfAttentiveWordExtractor":
            input_dim = params.pop_int("input_dim")
            params.assert_empty(cls.__name__)
            return SelfAttentiveWordExtractor(input_dim=input_dim)
