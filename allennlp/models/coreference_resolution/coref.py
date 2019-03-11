import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import overrides
from torch.autograd import Variable
from torch.nn.modules.linear import Linear

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder, SpanPruner, ConditionalRandomField
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import MentionRecall, ConllCorefScores
from allennlp.training.metrics import SpanBasedF1Measure

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("coref")
class CoreferenceResolver(Model):
    """
    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``text`` ``TextField`` we get as input to the model.
    context_layer : ``Seq2SeqEncoder``
        This layer incorporates contextual information for each word in the document.
    mention_feedforward : ``FeedForward``(线性层，计算候选spans的得分)
        This feedforward network is applied to the span representations which is then scored
        by a linear layer.
    antecedent_feedforward: ``FeedForward``（计算剪除后的spans对的得分，用于判断是否同指）
        This feedforward network is applied to pairs of span representation, along with any
        pairwise features, which is then scored by a linear layer.
    feature_size: ``int``（特征的个数）
        The embedding size for all the embedded features, such as distances or span widths.
    max_span_width: ``int``
        The maximum width of candidate spans.
    spans_per_word: float, required.
        A multiplier between zero and one which controls what percentage of candidate mention
        spans we retain with respect to the number of words in the document.
    max_antecedents: int, required.
        For each mention which survives the pruning stage, we consider this many antecedents.
        （修剪后剩余的spans都会作为antecendents考虑）
    label_namespace : ``str``, optional (default=``labels``)
        This is needed to compute the SpanBasedF1Measure metric.
        Unless you did something unusual, the default value should be what you want.
    constraint_type : ``str``, optional (default=``None``)
        If provided, the CRF will be constrained at decoding time
        to produce valid labels based on the specified type (e.g. "BIO", or "BIOUL").
    include_start_end_transitions : ``bool``, optional (default=``True``)
        Whether to include start and end transition parameters in the CRF.
    lexical_dropout: ``int``
        The probability of dropping out dimensions of the embedded text.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.（用于初始化模型参数）
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 context_layer: Seq2SeqEncoder,  # lstm
                 mention_feedforward: FeedForward,
                 antecedent_feedforward: FeedForward,
                 feature_size: int,
                 max_span_width: int,
                 spans_per_word: float,   #假设文档长度100，soans_per_word=0.4，那么就保留40个spans
                 max_antecedents: int,
                 label_namespace: str = "crf_span_labels",
                 constraint_type: str = "BIO",
                 include_start_end_transitions: bool = True,
                 lexical_dropout: float = 0.5,
                 dropout: float = 0.2,
                 features: Tuple[str] = ('pos_tags'),
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:  #正则化
        super().__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder # 字符向量和词向量


        feedforward_scorer = torch.nn.Sequential(
            TimeDistributed(mention_feedforward),
            TimeDistributed(torch.nn.Linear(mention_feedforward.get_output_dim(), 1)))

        self._mention_pruner = SpanPruner(feedforward_scorer)
        self._antecedent_feedforward = TimeDistributed(antecedent_feedforward)
        self._antecedent_scorer = TimeDistributed(torch.nn.Linear(antecedent_feedforward.get_output_dim(), 1))


        self._context_layer = context_layer
        self._endpoint_span_extractor = EndpointSpanExtractor(context_layer.get_output_dim(),
                                                              combination="x,y",
                                                              num_width_embeddings=max_span_width,
                                                              span_width_embedding_dim=feature_size,
                                                              bucket_widths=False)
        self._attentive_span_extractor = SelfAttentiveSpanExtractor(input_dim=context_layer.get_output_dim())

        self._num_distance_buckets = 10
        self._distance_embedding = Embedding(self._num_distance_buckets, feature_size)

        self._genre_embedding = Embedding(self.vocab.get_vocab_size("genres_tags"), feature_size)
        self._trigger_same_embedding = Embedding(2, feature_size)
        self._realies_same_embedding = Embedding(2, feature_size)

        self._features = features
        if "pos_tags" in features:
            self._pos_tag_embedding = Embedding(self.vocab.get_vocab_size("pos_tags"), feature_size)

        self._lexical_dropout = torch.nn.Dropout(lexical_dropout) if lexical_dropout > 0 else lambda x: x
        self._dropout = torch.nn.Dropout(dropout)

        self._max_span_width = max_span_width
        self._spans_per_word = spans_per_word
        self._max_antecedents = max_antecedents

        self.num_tags = self.vocab.get_vocab_size(label_namespace)
        self.tag_projection_layer = TimeDistributed(Linear(1220,
                                                           self.num_tags))

        if constraint_type is not None:
            # constraint_type = "BIO" or "BIOUL"
            # constraints = [List]
            labels = self.vocab.get_index_to_token_vocabulary(label_namespace)
            constraints = allowed_transitions(constraint_type, labels)
        else:
            constraints = None

        self.crf = ConditionalRandomField(
            self.num_tags, constraints,
            include_start_end_transitions = include_start_end_transitions
        )

        # 评测指标
        self._mention_recall = MentionRecall()
        self._conll_coref_scores = ConllCorefScores()

        self._span_metric = SpanBasedF1Measure(vocab,
                                             tag_namespace=label_namespace,
                                             label_encoding=constraint_type)

        self._loss_weight = torch.nn.Parameter(torch.FloatTensor([math.log(0.2), math.log(0.8)]))

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                text: Dict[str, torch.LongTensor],       #tokens
                spans: torch.IntTensor,
                genres_tags: torch.IntTensor,
                pos_tags: Optional[torch.IntTensor] = None,
                trigger_same_tags: Optional[torch.IntTensor] = None,
                realies_same_tags: Optional[torch.IntTensor] = None,
                span_labels: torch.IntTensor = None,      # tags
                crf_span_labels: torch.IntTensor = None,
                span_detection_labels: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        # (batch_size, document_length, embedding_size)
        text_embeddings = self._lexical_dropout(self._text_field_embedder(text)) # text (batch_size, document_length)
        document_length = text_embeddings.size(1)

        # Shape: (batch_size, document_length)
        text_mask = util.get_text_field_mask(text).float()

        feature_embeddings = list()
        if "pos_tags" in self._features:
            feature_embeddings.append(self._pos_tag_embedding(pos_tags))

        feature_embeddings = self._dropout(torch.cat(feature_embeddings, -1)) if feature_embeddings else None

        if feature_embeddings is not None:
            #1* *400
            contextualized_embeddings = self._dropout(self._context_layer(
                torch.cat([text_embeddings, feature_embeddings], -1), text_mask))
        else:
            contextualized_embeddings = self._dropout(self._context_layer(text_embeddings, text_mask))

        # (batch_size, num_spans, 2)
        num_spans = spans.size(1)
        # Shape: (batch_size, num_spans)   spans:(batch_size, num_spans, 2)
        span_mask = (spans[:, :, 0] >= 0).squeeze(-1).float()
         # Shape: (batch_size, num_spans, 2)
        spans = F.relu(spans.float()).long()

        # Shape: (batch_size, num_spans, 2 * encoding_dim + feature_size)
        endpoint_span_embeddings = self._endpoint_span_extractor(text_embeddings, contextualized_embeddings, spans)
        # Shape: (batch_size, num_spans, embedding_size)
        attended_span_embeddings = self._attentive_span_extractor(text_embeddings, contextualized_embeddings, spans)

        # Shape: (batch_size, num_spans, embedding_size + 2 * encoding_dim + feature_size)
        span_embeddings = torch.cat([endpoint_span_embeddings, attended_span_embeddings], -1)

        # Prune based on mention scores.
        # math.floor : 向下取整
        num_spans_to_keep = min(int(math.floor(self._spans_per_word * document_length)), spans.size(1))

        (top_span_embeddings, top_span_mask,
         top_span_indices, top_span_mention_scores ) = self._mention_pruner(span_embeddings,
                                                                            spans,
                                                                            span_mask,
                                                                            num_spans_to_keep)

        # top_span_mask.(batch_size, num_spans_to_keep)  ---> (batch_size, num_spans_to_keep, 1)
        top_span_mask = top_span_mask.unsqueeze(-1)
        # (batch_size*num_spans_to_keep)
        flat_top_span_indices = util.flatten_and_batch_shift_indices(top_span_indices, num_spans)
        # (batch_size, num_spans_to_keep, 2)
        top_spans = util.batched_index_select(spans,  #(batch_size, num_spans, 2)
                                              top_span_indices,   #(batch_size, num_spans_to_keep)
                                              flat_top_span_indices)   # (batch_size * num_spans_to_keep)

        #top_detection_spans = util.batched_index_select(span_detection_labels.unsqueeze(-1),
                                                        # top_span_indices,
                                                        # flat_top_span_indices)

        max_antecedents = min(self._max_antecedents, num_spans_to_keep)

        valid_antecedent_indices, valid_antecedent_offsets, valid_antecedent_log_mask = \
            self._generate_valid_antecedents(num_spans_to_keep, max_antecedents, util.get_device_of(text_mask))

        # Select tensors relating to the antecedent spans.
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        candidate_antecedent_embeddings = util.flattened_index_select(top_span_embeddings,
                                                                      valid_antecedent_indices)
        # Compute genre embedding.
        # (batch_size, num_genre(1), embedding_size)
        genre_embedding = self._genre_embedding(genres_tags)

        # trigger_same_type: Shape(batch_size, len_document) --> (batch_size, num_spans, 1)
        trigger_same_tags = util.batched_index_select(trigger_same_tags.unsqueeze(-1),
                                                      spans.split(1, dim=-1)[0].squeeze(-1))
        top_trigger_same_type = util.batched_index_select(trigger_same_tags, top_span_indices, flat_top_span_indices)

        candidate_antecedent_trigger_same_type = util.flattened_index_select(top_trigger_same_type,
                                                                             valid_antecedent_indices)

        candidate_anaphora_trigger_same_type = top_trigger_same_type.unsqueeze(-2).expand_as(
            candidate_antecedent_trigger_same_type)

        trigger_same_tags_agreement: torch.ByteTensor = candidate_antecedent_trigger_same_type == candidate_anaphora_trigger_same_type
        trigger_same_tags_agreement_embeddings = self._trigger_same_embedding(
            trigger_same_tags_agreement.squeeze(-1).long())

        realies_same_tags = util.batched_index_select(realies_same_tags.unsqueeze(-1),
                                                      spans.split(1, dim=-1)[0].squeeze(-1))
        top_realies_same_type = util.batched_index_select(realies_same_tags, top_span_indices, flat_top_span_indices)

        candidate_antecedent_realies_same_type = util.flattened_index_select(top_realies_same_type,
                                                                             valid_antecedent_indices)

        candidate_anaphora_realies_same_type = top_realies_same_type.unsqueeze(-2).expand_as(
            candidate_antecedent_realies_same_type)

        realies_same_tags_agreement: torch.ByteTensor = candidate_antecedent_realies_same_type == candidate_anaphora_realies_same_type
        realies_same_tags_agreement_embeddings = self._realies_same_embedding(
            realies_same_tags_agreement.squeeze(-1).long())

        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        span_pair_embeddings = self._compute_span_pair_embeddings(top_span_embeddings,
                                                                  candidate_antecedent_embeddings,
                                                                  genre_embedding,
                                                                  trigger_same_tags_agreement_embeddings,
                                                                  realies_same_tags_agreement_embeddings,
                                                                  valid_antecedent_offsets)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        candidate_antecedent_mention_scores = util.flattened_index_select(top_span_mention_scores,
                                                                          valid_antecedent_indices).squeeze(-1)
        # Shape: (batch_size, num_spans_to_keep, 1 + max_antecedents)
        coreference_scores = self._compute_coreference_scores(span_pair_embeddings,
                                                              top_span_mention_scores,
                                                              candidate_antecedent_mention_scores,
                                                              valid_antecedent_log_mask)

        # We now have, for each span which survived the pruning stage,
        # a predicted antecedent. This implies a clustering if we group
        # mentions which refer to each other in a chain.
        # Shape: (batch_size, num_spans_to_keep)
        _, predicted_antecedents = coreference_scores.max(2)  # 第二维最大的值，和最大的指的索引
        # Subtract one here because index 0 is the "no antecedent" class,
        # so this makes the indices line up with actual spans if the prediction
        # is greater than -1.
        predicted_antecedents -= 1


        output_dict = {"top_spans": top_spans,   #(batch_size, num_spans_to_keep, 2)
                       "antecedent_indices": valid_antecedent_indices,    # (num_spans_to_keep, max_antecedents),
                       "predicted_antecedents": predicted_antecedents,    # (batch_size, num_spans_to_keep)
                       }

        if span_labels is not None:
            # Find the gold labels for the spans which we kept.
            # Shape (batch_size, num_spans_to_keep, 1)
            pruned_gold_labels = util.batched_index_select(span_labels.unsqueeze(-1),  # Shape: (batch_size, num_spans, 1)
                                                           top_span_indices,  # Shape: (batch_size, num_spans_to_keep)
                                                           flat_top_span_indices) # Shape: (batch_size*num_spans_to_keep)
            # Shape (batch_size, num_spans_to_keep, max_spans_to_keep)
            antecedent_labels = util.flattened_index_select(pruned_gold_labels,
                                                            valid_antecedent_indices).squeeze(-1)
            antecedent_labels += valid_antecedent_log_mask.long()

            # Compute labels.
            # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
            gold_antecedent_labels = self._compute_antecedent_gold_labels(pruned_gold_labels,  # (btach_size, num_spans_to_keep, 1)
                                                                          antecedent_labels)    # (batch_size, num_spans_to_keep, max_spans_to_keep)
            #
            coreference_log_probs = util.last_dim_log_softmax(coreference_scores, top_span_mask)
            correct_antecedent_log_probs = coreference_log_probs + gold_antecedent_labels.log()
            negative_marginal_log_likelihood = -util.logsumexp(correct_antecedent_log_probs).sum()


            self._mention_recall(top_spans, metadata)
            self._conll_coref_scores(top_spans, valid_antecedent_indices, predicted_antecedents, metadata)


        # 将Bi-LSTM输入到模型tag_projection_layer中，得到输出logits(未归一化的概率)
        # 再将logits输入到crf模型中，计算输出tags
        # (batch_size, document_length, num_tags)
        logits = self.tag_projection_layer(span_embeddings)
        # 预测的时候用维比特算法
        event_detection_predicted_tags = self.crf.viterbi_tags(logits, span_mask)

        output_dict["logits"] = logits
        output_dict["span_mask"] = span_mask
        output_dict["event_detection_predicted_tags"] = event_detection_predicted_tags


        if crf_span_labels is not None:
            # Add negative log-likelihood as loss
            log_likelihood = self.crf(logits, crf_span_labels, span_mask)

            # Represent viterbi tags as "class probabilities" that we can
            # feed into the metrics
            class_probabilities = logits * 0.
            for i, instance_tags in enumerate(event_detection_predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1

            self._span_metric(class_probabilities,crf_span_labels, span_mask.float())


        weight = F.softmax(self._loss_weight, 0)
        output_dict["loss"] = weight[1] * negative_marginal_log_likelihood \
                              - weight[0] * log_likelihood \


        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):


        # A tensor of shape (batch_size, num_spans_to_keep, 2), representing
        # the start and end indices of each span.
        batch_top_spans = output_dict["top_spans"].data.cpu()

        # A tensor of shape (batch_size, num_spans_to_keep) representing, for each span,
        # the index into ``antecedent_indices`` which specifies the antecedent span. Additionally,
        # the index can be -1, specifying that the span has no predicted antecedent.
        batch_predicted_antecedents = output_dict["predicted_antecedents"].data.cpu()

        # A tensor of shape (num_spans_to_keep, max_antecedents), representing the indices
        # of the predicted antecedents with respect to the 2nd dimension of ``batch_top_spans``
        # for each antecedent we considered.
        antecedent_indices = output_dict["antecedent_indices"].data.cpu()
        batch_clusters: List[List[List[Tuple[int, int]]]] = []

        # Calling zip() on two tensors results in an iterator over their
        # first dimension. This is iterating over instances in the batch.
        for top_spans, predicted_antecedents in zip(batch_top_spans, batch_predicted_antecedents):
            spans_to_cluster_ids: Dict[Tuple[int, int], int] = {}
            clusters: List[List[Tuple[int, int]]] = []

            for i, (span, predicted_antecedent) in enumerate(zip(top_spans, predicted_antecedents)):
                if predicted_antecedent < 0:
                    # We don't care about spans which are
                    # not co-referent with anything.
                    continue


                predicted_index = antecedent_indices[i, predicted_antecedent]

                antecedent_span = (top_spans[predicted_index, 0],
                                   top_spans[predicted_index, 1])
                # Check if we've seen the span before.
                if antecedent_span in spans_to_cluster_ids.keys():
                    predicted_cluster_id: int = spans_to_cluster_ids[antecedent_span]
                else:
                    # We start a new cluster.
                    predicted_cluster_id = len(clusters)
                    # Append a new cluster containing only this span.
                    clusters.append([antecedent_span])
                    # Record the new id of this span.
                    spans_to_cluster_ids[antecedent_span] = predicted_cluster_id

                # Now add the span we are currently considering.
                span_start, span_end = span
                clusters[predicted_cluster_id].append((span_start, span_end))
                spans_to_cluster_ids[(span_start, span_end)] = predicted_cluster_id
            batch_clusters.append(clusters)

        output_dict["clusters"] = batch_clusters
        return output_dict

    @overrides
    # 评估
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        Metrics = {}
        mention_recall = self._mention_recall.get_metric(reset)
        result = self._conll_coref_scores.get_metric(reset)
        coref_P, coref_R, coref_f1 = result['avg']
        # metric_dict = self._span_metric.get_metric(reset=reset)
        # for x, y in metric_dict.items():
        #     if "overall" in x:
        #         Metrics[x] = y
        Metrics["mention_recall"] = mention_recall
        Metrics["coref_precison"] = coref_P
        Metrics["coref_recall"] = coref_R
        Metrics["coref_f1"] = coref_f1
        return Metrics

    @staticmethod
    def _generate_valid_antecedents(num_spans_to_keep: int,
                                    max_antecedents: int,
                                    device: int) -> Tuple[torch.IntTensor,
                                                          torch.IntTensor,
                                                          torch.FloatTensor]:

        # Shape: (num_spans_to_keep, 1)[0,1,2,....]
        target_indices = util.get_range_vector(num_spans_to_keep, device).unsqueeze(1)

        # Shape: (1, max_antecedents)[1,2,3..]
        valid_antecedent_offsets = (util.get_range_vector(max_antecedents, device) + 1).unsqueeze(0)

        # This is a broadcasted subtraction. 广播减法
        # Shape: (num_spans_to_keep, max_antecedents)
        raw_antecedent_indices = target_indices - valid_antecedent_offsets


        valid_antecedent_log_mask = (raw_antecedent_indices >= 0).float().unsqueeze(0).log()

        # Shape: (num_spans_to_keep, max_antecedents) relu = max(0, x)
        valid_antecedent_indices = F.relu(raw_antecedent_indices.float()).long()
        return valid_antecedent_indices, valid_antecedent_offsets, valid_antecedent_log_mask

    def _compute_span_pair_embeddings(self,
                                      top_span_embeddings: torch.FloatTensor,
                                      antecedent_embeddings: torch.FloatTensor,
                                      genre_embedding: torch.FloatTensor,
                                      trigger_same_type_agreement_embeddings: torch.FloatTensor,
                                      realies_same_type_agreement_embeddings: torch.FloatTensor,
                                      antecedent_offsets: torch.FloatTensor):
        """
        Computes an embedding representation of pairs of spans for the pairwise scoring function
        to consider. This includes both the original span representations, the element-wise
        similarity of the span representations, and an embedding representation of the distance
        between the two spans.

        Parameters
        ----------
        top_span_embeddings : ``torch.FloatTensor``, required.
            Embedding representations of the top spans. Has shape
            (batch_size, num_spans_to_keep, embedding_size).
        antecedent_embeddings : ``torch.FloatTensor``, required.
            Embedding representations of the antecedent spans we are considering
            for each top span. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, embedding_size).
        antecedent_offsets : ``torch.IntTensor``, required.
            The offsets between each top span and its antecedent spans in terms
            of spans we are considering. Has shape (1, max_antecedents).

        Returns
        -------
        span_pair_embeddings : ``torch.FloatTensor``
            Embedding representation of the pair of spans to consider. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        """

        # Shape: (1, max_antecedents, embedding_size)
        antecedent_distance_embeddings = self._distance_embedding(
            util.bucket_values(antecedent_offsets,
                               num_total_buckets=self._num_distance_buckets))

        # Shape: (1, 1, max_antecedents, embedding_size)
        antecedent_distance_embeddings = antecedent_distance_embeddings.unsqueeze(0)

        expanded_distance_embeddings_shape = (antecedent_embeddings.size(0),  # batch_size
                                              antecedent_embeddings.size(1),  # num_spans_to_keep
                                              antecedent_embeddings.size(2),  # max_antecedents
                                              antecedent_distance_embeddings.size(-1))  # embedding_size
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        antecedent_distance_embeddings = antecedent_distance_embeddings.expand(*expanded_distance_embeddings_shape)
        antecedent_genre_embeddings = genre_embedding.view(1, 1, 1, -1).expand_as(antecedent_distance_embeddings)
        feature_embeddings = self._dropout(torch.cat(
            [antecedent_genre_embeddings, realies_same_type_agreement_embeddings, trigger_same_type_agreement_embeddings],-1))
            # ], -1))

            # [antecedent_distance_embeddings, antecedent_genre_embeddings, trigger_same_type_agreement_embeddings,
            #  realies_same_type_agreement_embeddings], -1))
        # feature_embeddings = self._dropout(torch.cat(
        #     [antecedent_genre_embeddings, trigger_same_type_agreement_embeddings], -1
        # ))

        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        target_embeddings = top_span_embeddings.unsqueeze(2).expand_as(antecedent_embeddings)
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        span_pair_embeddings = torch.cat([target_embeddings,
                                          antecedent_embeddings,
                                          antecedent_embeddings * target_embeddings,
                                          feature_embeddings], -1)
        return span_pair_embeddings

    @staticmethod
    def _compute_antecedent_gold_labels(top_span_labels: torch.IntTensor,
                                        antecedent_labels: torch.IntTensor):
        """
        Generates a binary indicator for every pair of spans. This label is one if and
        only if the pair of spans belong to the same cluster. The labels are augmented
        with a dummy antecedent at the zeroth position, which represents the prediction
        that a span does not have any antecedent.

        Parameters
        ----------
        top_span_labels : ``torch.IntTensor``, required.
            The cluster id label for every span. The id is arbitrary,
            as we just care about the clustering. Has shape (batch_size, num_spans_to_keep).
        antecedent_labels : ``torch.IntTensor``, required.
            The cluster id label for every antecedent span. The id is arbitrary,
            as we just care about the clustering. Has shape
            (batch_size, num_spans_to_keep, max_antecedents).

        Returns
        -------
        pairwise_labels_with_dummy_label : ``torch.FloatTensor``
            A binary tensor representing whether a given pair of spans belong to
            the same cluster in the gold clustering.
            Has shape (batch_size, num_spans_to_keep, max_antecedents + 1).

        """
        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        target_labels = top_span_labels.expand_as(antecedent_labels)
        same_cluster_indicator = (target_labels == antecedent_labels).float()
        non_dummy_indicator = (target_labels >= 0).float()
        pairwise_labels = same_cluster_indicator * non_dummy_indicator

        # Shape: (batch_size, num_spans_to_keep, 1)
        dummy_labels = (1 - pairwise_labels).prod(-1, keepdim=True)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
        pairwise_labels_with_dummy_label = torch.cat([dummy_labels, pairwise_labels], -1)
        return pairwise_labels_with_dummy_label

    def _compute_coreference_scores(self,
                                    pairwise_embeddings: torch.FloatTensor,
                                    top_span_mention_scores: torch.FloatTensor,
                                    antecedent_mention_scores: torch.FloatTensor,
                                    antecedent_log_mask: torch.FloatTensor) -> torch.FloatTensor:

        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        antecedent_scores = self._antecedent_scorer(
            self._antecedent_feedforward(pairwise_embeddings)).squeeze(-1)
        antecedent_scores += top_span_mention_scores + antecedent_mention_scores
        antecedent_scores += antecedent_log_mask

        # Shape: (batch_size, num_spans_to_keep, 1)
        shape = [antecedent_scores.size(0), antecedent_scores.size(1), 1]
        dummy_scores = Variable(antecedent_scores.data.new(*shape).fill_(0), requires_grad=False)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
        coreference_scores = torch.cat([dummy_scores, antecedent_scores], -1)
        return coreference_scores

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> "CoreferenceResolver":
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        context_layer = Seq2SeqEncoder.from_params(params.pop("context_layer"))
        mention_feedforward = FeedForward.from_params(params.pop("mention_feedforward"))
        antecedent_feedforward = FeedForward.from_params(params.pop("antecedent_feedforward"))

        feature_size = params.pop_int("feature_size")
        max_span_width = params.pop_int("max_span_width")
        spans_per_word = params.pop_float("spans_per_word")
        max_antecedents = params.pop_int("max_antecedents")
        lexical_dropout = params.pop_float("lexical_dropout", 0.5)
        dropout = params.pop_float("dropout", 0.2)
        features = tuple(params.pop("features", []))
        label_namespace = params.pop("label_namespace", "crf_span_labels")
        constraint_type = params.pop("constraint_type", None)
        include_start_end_transitions = params.pop("include_start_end_transitions", True)
        init_params = params.pop("initializer", None)
        reg_params = params.pop("regularizer", None)
        initializer = (InitializerApplicator.from_params(init_params)
                       if init_params is not None
                       else InitializerApplicator())
        regularizer = RegularizerApplicator.from_params(reg_params) if reg_params is not None else None

        params.assert_empty(cls.__name__)
        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   context_layer=context_layer,
                   mention_feedforward=mention_feedforward,
                   antecedent_feedforward=antecedent_feedforward,
                   feature_size=feature_size,
                   max_span_width=max_span_width,
                   spans_per_word=spans_per_word,
                   max_antecedents=max_antecedents,
                   lexical_dropout=lexical_dropout,
                   dropout = dropout,
                   features=features,
                   label_namespace=label_namespace,
                   constraint_type=constraint_type,
                   include_start_end_transitions=include_start_end_transitions,
                   initializer=initializer,
                   regularizer=regularizer)