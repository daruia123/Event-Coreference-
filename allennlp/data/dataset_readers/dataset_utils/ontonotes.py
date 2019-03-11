from typing import DefaultDict, List, Optional, Iterator, Set, Tuple
from collections import defaultdict
import codecs
import logging

from nltk import Tree, ParentedTree

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

TypedSpan = Tuple[int, Tuple[int, int]]  # pylint: diasble=invalid-name
TypedStringSpan = Tuple[str, Tuple[int, int]]  # pylint: disable=invalid-name


class OntontesSentence:
    def __init__(self,
                 document_id: str,
                 words: List[str],
                 pos_tags: List[str],
                 parse_tree: Optional[Tree],
                 subtype_same: List[Tuple],
                 realies_same: List[Tuple],
                 coref_spans: Set[TypedSpan],
                 span_crf: List[Tuple]) -> None:
        self.document_id = document_id
        self.words = words
        self.pos_tags = pos_tags
        self.parse_tree = parse_tree
        self.subtype_same = subtype_same
        self.realies_same = realies_same
        self.coref_spans = coref_spans
        self.span_crf = span_crf


class Ontonotes:
    def dataset_document_iterator(self, file_path: str) -> Iterator[List[OntontesSentence]]:

        logger.info("Reading format sentences from dataset files at: %s", file_path)
        with codecs.open(file_path, 'r') as open_file:
            conll_rows = []
            document: List[OntontesSentence] = []
            for line in open_file:
                line = line.strip()
                if line != '' and not line.startswith('#'):
                    conll_rows.append(line)
                else:
                    if conll_rows:
                        sentence = self._conll_rows_to_sentence(conll_rows)

                        if sentence:
                            document.append(sentence)
                        conll_rows = []
                if line.startswith('#end document'):
                    yield document
                    document = []
                    if document:
                        yield document

    def _conll_rows_to_sentence(self, conll_rows: List[str]) -> OntontesSentence:
        document_id: str = None
        Word: List[str] = []
        pos_tags: List[str] = []
        parse_pieces: List[str] = []
        subtype_same_label: List[Tuple] = []
        realies_same_label: List[Tuple] = []
        current_stack = None
        current_label = None

        clusters: DefaultDict[int, List[Tuple[int, int]]] = defaultdict(list)
        span_crf: DefaultDict[str, List[Tuple[int, int]]] = defaultdict(list)
        coref_stacks: DefaultDict[int, List[int]] = defaultdict(list)

        for index, row in enumerate(conll_rows):

            conll_components = row.split('\t')
            if len(conll_components) > 1:
                document_id = conll_components[0]
                word = conll_components[2]
                pos_tag = conll_components[3]
                parse_piece = conll_components[4]
                parse_word = word

                num = parse_piece.count('*')
                if num == 2:
                    parse_piece = parse_piece.replace('*', word, 1)
                elif num > 2:
                    parse_piece = parse_piece.replace('**', word, 1)

                (left_brackets, right_hand_side) = parse_piece.split('*')
                right_brackets = right_hand_side.count(')') * ')'
                parse_piece = f'{left_brackets}({pos_tag} {parse_word}) {right_brackets}'

                self._process_coref_span_annotations_for_word(conll_components[-1], index, clusters, span_crf, coref_stacks)
                # current_stack = self._process_coref_span_annotations_for_word(conll_components[-1], conll_components[-3], index, clusters, span_crf,
                #                                               coref_stacks, current_stack)
                self._process_realies_arg_for_word(conll_components[-2], index, realies_same_label)
                current_label = self._process_trigger_for_word(conll_components[5], conll_components[-1], index,
                                                            current_label, subtype_same_label)


                Word.append(word)
                pos_tags.append(pos_tag)
                parse_pieces.append(parse_piece)

        parse_tree = ParentedTree.fromstring("".join(parse_pieces))
        coref_span_tuples: Set[TypedSpan] = {(cluster_id, span)
                                             for cluster_id, span_list in clusters.items()
                                             for span in span_list}
        span_crf_tuples = [(span_crf_id, span)
                                           for span_crf_id, span_list in span_crf.items()
                                           for span in span_list]

        return OntontesSentence(document_id,
                                Word,
                                pos_tags,
                                parse_tree,
                                subtype_same_label,
                                realies_same_label,
                                coref_span_tuples,
                                span_crf_tuples)

    @staticmethod
    def _process_coref_span_annotations_for_word(label: str,
                                                 word_index: int,
                                                 clusters: DefaultDict[int, List[Tuple[int, int]]],
                                                 span_crf: DefaultDict[int, List[Tuple[int, int]]],
                                                 coref_stacks: DefaultDict[int, List[int]]) -> None:
        if label != "-":
            if label[0] == '(':
                if label[-1] == ')':
                    if label[1] != '-':
                        cluster_id = int(label[1])
                        span_crf_id = 'B-C'
                        clusters[cluster_id].append((word_index, word_index))
                        span_crf[span_crf_id].append((word_index, word_index))
                    elif label[1] == '-':
                        span_crf_id = 'B-S'
                        span_crf[span_crf_id].append((word_index, word_index))
                else:
                    if label[1] != '-':
                        cluster_id = int(label[1])
                        coref_stacks[cluster_id].append(word_index)
                    elif label[1] == '-':
                        span_crf_id = 'B-S'
                        coref_stacks[span_crf_id].append(word_index)
            else:
                if label[0] != '-':
                    cluster_id = int(label[0])
                    span_crf_id = 'B-C'
                    if coref_stacks[cluster_id] != []:
                        start = coref_stacks[cluster_id].pop()
                        clusters[cluster_id].append((start, word_index))
                        span_crf[span_crf_id].append((start, word_index))
                elif label[0] == '-':
                    span_crf_id = 'B-S'
                    if coref_stacks[span_crf_id] != []:
                        start = coref_stacks[span_crf_id].pop()
                        span_crf[span_crf_id].append((start, word_index))

    # @staticmethod
    # def _process_coref_span_annotations_for_word(label: str,
    #                                              event_label: str,
    #                                              word_index: int,
    #                                              clusters: DefaultDict[int, List[Tuple[int, int]]],
    #                                              span_crf: DefaultDict[str, List[Tuple[int, int]]],
    #                                              coref_stacks: DefaultDict[int, List[int]] ,
    #                                              current_stacks) -> None:
    #     if label != "-":
    #         if label[0] == '(':
    #             if label[-1] == ')':
    #                 if label[1] != '-':
    #                     cluster_id = int(label[1])
    #                     span_crf_id = 'B' + event_label.strip('()*')
    #                     clusters[cluster_id].append((word_index, word_index))
    #                     span_crf[span_crf_id].append((word_index, word_index))
    #                 elif label[1] == '-':
    #                     span_crf_id = 'B' + event_label.strip('()*')
    #                     span_crf[span_crf_id].append((word_index, word_index))
    #             else:
    #                 if label[1] != '-':
    #                     cluster_id = int(label[1])
    #                     span_crf_id = 'B' + event_label.strip('()*')
    #                     current_stacks = span_crf_id
    #                     coref_stacks[span_crf_id].append(word_index)
    #                     coref_stacks[cluster_id].append(word_index)
    #                 elif label[1] == '-':
    #                     span_crf_id = 'B' + event_label.strip('()*')
    #                     current_stacks = span_crf_id
    #                     coref_stacks[span_crf_id].append(word_index)
    #         else:
    #             if label[0] != '-':
    #                 cluster_id = int(label[0])
    #                 span_crf_id = current_stacks
    #                 if coref_stacks[cluster_id] != []:
    #                     start = coref_stacks[cluster_id].pop()
    #                     clusters[cluster_id].append((start, word_index))
    #                     span_crf[span_crf_id].append((start, word_index))
    #             elif label[0] == '-':
    #                 span_crf_id = current_stacks
    #                 if coref_stacks[span_crf_id] != []:
    #                     start = coref_stacks[span_crf_id].pop()
    #                     span_crf[span_crf_id].append((start, word_index))
    #
    #     return current_stacks



    @staticmethod
    def _process_trigger_for_word(trigger_word: str,
                                  coref_word: str,
                                  index: int,
                                  current_label: None,
                                  subtype_same_label: List[Tuple]) -> None:

        label = trigger_word.strip('()*')
        if "(" in trigger_word:
            bo_label = label
            subtype_same_label.append((index, bo_label))
            current_label = label

        elif current_label != None:
            bo_label = current_label
            subtype_same_label.append((index, bo_label))

        else:
            subtype_same_label.append((index, str(index)))

        if ")" in trigger_word:
            current_label = None

        return current_label


    @staticmethod
    def _process_realies_arg_for_word(mut_label: str,
                                      word_index: int,
                                      realies_same_label: List[Tuple]) -> None:
        realies = mut_label
        if realies == '-':

            realies_same_label.append((word_index, str(word_index)))
        if realies != '-':
            label_realies = realies.strip('()')


            realies_same_label.append((word_index, label_realies))


