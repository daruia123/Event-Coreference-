import logging
import collections
from typing import Any, Dict, List, Optional, Tuple, DefaultDict, Set

from nltk import Tree, ParentedTree

from overrides import overrides

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, ListField, TextField, SpanField, MetadataField, SequenceLabelField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, enumerate_spans, OntontesSentence

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def canonicalize_clusters(clusters: DefaultDict[int, List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:

    merged_clusters: List[Set[Tuple[int, int]]] = []
    merged_crf_spans: List[Set[Tuple[int, int]]] = []
    for cluster in clusters.values():
        merged_clusters.append(set(cluster))


    return [list(c) for c in merged_clusters]


@DatasetReader.register("coref")
class ConllCorefReader(DatasetReader):
    """
    Reads a single CoNLL-formatted file. This is the same file format as used in the
    :class:`~allennlp.data.dataset_readers.semantic_role_labelling.SrlReader`, but is preprocessed
    to dump all documents into a single file per train, dev and test split. See
    scripts/compile_coref_data.sh for more details of how to pre-process the Ontonotes 5.0 data
    into the correct format.

    Returns a ``Dataset`` where the ``Instances`` have four fields: ``text``, a ``TextField``
    containing the full document text, ``spans``, a ``ListField[SpanField]`` of inclusive start and
    end indices for span candidates, and ``metadata``, a ``MetadataField`` that stores the instance's
    original text. For data with gold cluster labels, we also include the original ``clusters``
    (a list of list of index pairs) and a ``SequenceLabelField`` of cluster ids for every span
    candidate.

    Parameters
    ----------
    max_span_width: ``int``, required.
        The maximum width of candidate spans to consider.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        This is used to index the words in the document.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 max_span_width: int,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 features: Tuple[str] = ('pos_tags', 'trigger_tags', 'realies_tags'),
                 max_node_depth: int = 0,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._features = tuple(features)
        self._max_node_depth = max_node_depth

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = file_path

        ontonotes_reader = Ontonotes()
        for sentences in ontonotes_reader.dataset_document_iterator(file_path):
            clusters: DefaultDict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
            crf_spans: DefaultDict[str, List[Tuple[int, int]]] = collections.defaultdict(list)

            total_tokens = 0
            for sentence in sentences:
                for typed_span in sentence.coref_spans:
                    span_id, (start, end) = typed_span
                    clusters[span_id].append((start + total_tokens,
                                              end + total_tokens))

                for crf_span_type in sentence.span_crf:
                    crf_span_id, (crf_start, crf_end) = crf_span_type
                    crf_spans[crf_span_id].append((crf_start+ total_tokens,
                                                   crf_end+total_tokens))
                total_tokens += len(sentence.words)

            canonical_clusters  = canonicalize_clusters(clusters)
            yield self.text_to_instance(sentences, canonical_clusters, crf_spans)

    @overrides
    def text_to_instance(self,  # type: ignore
                         sentences: List[OntontesSentence],
                         gold_clusters: Optional[List[List[Tuple[int, int]]]] = None,
                         gold_crf_spans: DefaultDict[str, List[Tuple[int, int]]] = None) -> Instance:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        sentences : ``List[List[str]]``, required.
            A list of lists representing the tokenised words and sentences in the document.
        gold_clusters : ``Optional[List[List[Tuple[int, int]]]]``, optional (default = None)
            A list of all clusters in the document, represented as word spans. Each cluster
            contains some number of spans, which can be nested and overlap, but will never
            exactly match between clusters.

        Returns
        -------
        An ``Instance`` containing the following ``Fields``:
            text : ``TextField``
                The text of the full document.
            spans : ``ListField[SpanField]``
                A ListField containing the spans represented as ``SpanFields``
                with respect to the document text.
            span_labels : ``SequenceLabelField``, optional
                The id of the cluster which each possible span belongs to, or -1 if it does
                 not belong to a cluster. As these labels have variable length (it depends on
                 how many spans we are considering), we represent this a as a ``SequenceLabelField``
                 with respect to the ``spans ``ListField``.
        """
        flattened_sentences = [self._normalize_word(word) for sentence in sentences for word in sentence.words]
        words = [Token(word) for word in flattened_sentences]
        text_field = TextField(words, self._token_indexers)  # text_field-->TextField

        genre_tags = sentences[0].document_id[:2] # genre = str
        genre_field = LabelField(genre_tags, label_namespace="genres_tags")  # 给"genres"namespace一个id

        trigger_same_tags = [trigger[1] for sentence in sentences for trigger in sentence.subtype_same]
        trigger_same_field = SequenceLabelField(trigger_same_tags, text_field, label_namespace="trigger_same_tags")

        realies_same_tags = [realies[1] for sentence in sentences for realies in sentence.realies_same]
        realies_same_field = SequenceLabelField(realies_same_tags, text_field, label_namespace="realies_same_tags")

        # spans(int ,int)   span_labels用于coreference_resolution  span_detection_label用于coreference detection
        spans, span_labels, span_detection_labels, crf_span_labels = self._get_spans_and_labels(sentences, text_field,
                                                                               gold_clusters, gold_crf_spans) # spans是SpanField

        metadata: Dict[str, Any] = {"original_text": flattened_sentences}
        if gold_clusters is not None:
            metadata["clusters"] = gold_clusters


        span_field = ListField(spans)
        metadata_field = MetadataField(metadata)

        fields: Dict[str, Field] = {"text": text_field,
                                    "genres_tags": genre_field,
                                    "trigger_same_tags": trigger_same_field,
                                    "realies_same_tags": realies_same_field,
                                    "spans": span_field,
                                    "metadata": metadata_field,
                                    }
        if span_labels is not None:
            fields["span_labels"] = SequenceLabelField(span_labels, span_field)
        if span_detection_labels is not None:
            fields["span_detection_labels"] = SequenceLabelField(span_detection_labels, span_field)
        if crf_span_labels is not None:
            fields["crf_span_labels"] = SequenceLabelField(crf_span_labels, span_field, label_namespace="crf_span_labels")

        if "pos_tags" in self._features:
            pos_tags = [pos_tag for sentence in sentences for pos_tag in sentence.pos_tags]
            pos_tag_field = SequenceLabelField(pos_tags, text_field, label_namespace="pos_tags")

            metadata["pos_tags"] = pos_tags
            fields["pos_tags"] = pos_tag_field

        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> "ConllCorefReader":
        token_indexers = TokenIndexer.dict_from_params(params.pop("token_indexers", {}))
        max_span_width = params.pop_int("max_span_width")
        features = tuple(params.pop('features', []))
        lazy = params.pop('lazy', False)
        params.assert_empty(cls.__name__)
        return cls(token_indexers=token_indexers, max_span_width=max_span_width,
                   features=features,lazy=lazy)


    @staticmethod
    def _normalize_word(word):
        if word == "/." or word == "/?":
            return word[1:]
        else:
            return word

    def _get_spans_and_labels(self, sentences: List[OntontesSentence],
                              text_field: TextField,
                              gold_clusters: Optional[List[List[Tuple[int, int]]]] = None,
                              gold_crf_spans = None):
        cluster_dict = {}
        if gold_clusters is not None:
            for cluster_id, cluster in enumerate(gold_clusters):
                for mention in cluster:
                    cluster_dict[tuple(mention)] = cluster_id

        crf_span_dict = {}
        if gold_crf_spans is not None:
            for crf_span_id, spans in gold_crf_spans.items():
                for span in spans:
                    crf_span_dict[span] = crf_span_id

        spans: List[Field] = []
        spans_labels: Optional[List[int]] = [] if gold_clusters is not None else None
        crf_span_labels: Optional[List[str]] = [] if gold_crf_spans is not None else None
        spans_detection_labels: Optional[List[int]] = [] if gold_clusters is not None else None

        sentence_offset = 0
        for sentence in sentences:
            for start, end in enumerate_spans(sentence.words,
                                              sentence.pos_tags,
                                              offset=sentence_offset,
                                              max_span_width=self._max_span_width):
                if spans_labels is not None:
                    if (start, end) in cluster_dict:
                        spans_labels.append(cluster_dict[(start, end)])
                        spans_detection_labels.append(1)
                    else:
                        spans_labels.append(-1)
                        spans_detection_labels.append(0)

                if crf_span_labels is not None:
                    if (start, end) in crf_span_dict:
                        crf_span_labels.append(crf_span_dict[(start, end)])
                    else:
                        crf_span_labels.append('O')
                spans.append(SpanField(start, end, text_field))
            sentence_offset += len(sentence.words)

        return spans,spans_labels, spans_detection_labels, crf_span_labels


    @staticmethod
    def _get_all_root_to_leaf_paths(root: Tree):

        def traverse(tree: Tree, cache_path: str, paths: List[List[str]]):
            if not (tree and isinstance(tree, Tree)):
                return []
            cache_path += tree.label()
            if len(tree) == 1 and not isinstance(tree[0], Tree):
                paths.append(cache_path.split(" "))
            else:
                for child in tree:
                    traverse(child, cache_path + " ", paths)

        path = ""
        all_paths = list()
        traverse(root, path, all_paths)
        return all_paths
