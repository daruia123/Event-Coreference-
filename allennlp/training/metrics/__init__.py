"""
A :class:`~allennlp.training.metrics.metric.Metric` is some quantity or quantities
that can be accumulated during training or evaluation; for example,
accuracy or F1 score.
"""

from allennlp.training.metrics.span_based_f1_measure import SpanBasedF1Measure
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy
from allennlp.training.metrics.conll_coref_scores import ConllCorefScores
from allennlp.training.metrics.mention_recall import MentionRecall
from allennlp.training.metrics.metric import Metric
