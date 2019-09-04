from typing import Dict, Sequence, List, Union
import logging

import jsonlines
import numpy as np
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, NamespaceSwappingField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer


logger = logging.getLogger(__name__)


@DatasetReader.register('cpae_reader')
class CPAEReader(DatasetReader):
    """
    We use different namespace for word and definition.
    When using, we map word from word namespace to definition namespace to extract the same representation.
    """
    def __init__(
        self,
        definition_tokenizer: Tokenizer = None,
        word_tokenizer: Tokenizer = None,
        definition_token_indexers: Dict[str, TokenIndexer] = None,
        word_token_indexers: Dict[str, TokenIndexer] = None,
        definition_namespace: str = 'definition',
        word_namespace: str = 'word',
        max_definition_length: int = None,
        lazy: bool = False
    ) -> None:

        super().__init__(lazy=lazy)
        self.max_definition_length = max_definition_length

        self._definition_namespace = definition_namespace
        self._word_namespace = word_namespace

        self._definition_tokenizer = definition_tokenizer or WordTokenizer()
        self._word_tokenizer = word_tokenizer

        self._definition_token_indexers = definition_token_indexers or \
            {'tokens': SingleIdTokenIndexer(namespace=self._definition_namespace)}
        self._word_token_indexers = word_token_indexers or \
            {"tokens": SingleIdTokenIndexer(namespace=self._word_namespace)}

    @overrides
    def _read(self, file_path):
        with jsonlines.open(cached_path(file_path)) as reader:
            for line in reader:
                yield self.text_to_instance(line['definition'], line['word'])

    @overrides
    def text_to_instance(self, definition: Union[str, List[str]], word: str = None) -> Instance:

        if self._definition_tokenizer is None or isinstance(definition, List):
            tokenized_definition = [Token(w) for w in definition]
        else:
            tokenized_definition = self._definition_tokenizer.tokenize(definition)
        if self.max_definition_length is not None:
            tokenized_definition = tokenized_definition[:self.max_definition_length]
        definition_field = TextField(tokenized_definition, self._definition_token_indexers)
        fields_dict = {'definition': definition_field}

        if word is not None:
            if self._word_tokenizer is None:
                tokenized_word = [Token(word)]
            else:
                tokenized_word = self._word_tokenizer.tokenize(word)
            word_field = TextField(tokenized_word, self._word_token_indexers)
            fields_dict['word'] = word_field

            # mapping words from word_namespace to definition_namespace
            word_to_definition_field = NamespaceSwappingField(tokenized_word, self._definition_namespace)
            fields_dict['word_to_definition'] = word_to_definition_field

        return Instance(fields_dict)
