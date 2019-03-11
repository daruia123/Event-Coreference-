"""
A :class:`~allennlp.data.dataset_readers.dataset_reader.DatasetReader`
reads a file and converts it to a
:class:`~allennlp.data.dataset.Dataset`.
The various subclasses know how to read specific filetypes
and produce datasets in the formats required by specific models.
"""

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.coreference_resolution import ConllCorefReader