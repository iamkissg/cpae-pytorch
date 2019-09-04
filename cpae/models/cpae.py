from typing import Dict, Optional
import logging

from overrides import overrides
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss

from allennlp.common.checks import ConfigurationError, check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util

from cpae.training.metrics import EuclideanDistance

logger = logging.getLogger(__name__)


@Model.register('cpae')
class CPAE(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_embedder: TextFieldEmbedder,
        definition_encoder: Seq2SeqEncoder,
        definition_decoder: FeedForward,
        definition_feedforward: FeedForward = None,
        definition_pooling: str = 'last',
        definition_namespace: str = 'definition',
        word_namespace: str = 'word',
        alpha: float = 1.0,
        beta: float = 8.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super().__init__(vocab, regularizer)

        self.definition_namespace = definition_namespace
        self.word_namespace = word_namespace
        self.definition_vocab_size = self.vocab.get_vocab_size(namespace=self.definition_namespace)
        self._oov_index = self.vocab.get_token_index(self.vocab._oov_token, self.definition_namespace)
        self.limited_word_vocab_size = None

        self.alpha = alpha
        self.beta = beta
        self.eps = 10e-8

        logger.info(f'Definition vocab size: {self.vocab.get_vocab_size(namespace=self.definition_namespace)}')
        logger.info(f'Word vocab size: {self.vocab.get_vocab_size(namespace=self.word_namespace)}')
        logger.info('Intersection vocab size: {}'.format(
            len(set(self.vocab._token_to_index[self.definition_namespace].keys())
                .intersection(set(self.vocab._token_to_index[self.word_namespace].keys())))))

        # TODO: check text_embedder
        self.text_embedder = text_embedder
        self.definition_encoder = definition_encoder
        self.definition_decoder = definition_decoder
        self.definition_pooling = definition_pooling
        if definition_feedforward is not None:
            self.definition_feedforward = definition_feedforward
        else:
            self.definition_feedforward = lambda x: x
        if self.definition_pooling == 'self-attentive':
            self.self_attentive_pooling_projection = nn.Linear(
                self.definition_encoder.get_output_dim(), 1)

        # checks
        check_dimensions_match(text_embedder.get_output_dim(), definition_encoder.get_input_dim(),
                               'emb_dim', 'encoder_input_dim')
        if self.definition_decoder.get_output_dim() > self.vocab.get_vocab_size(definition_namespace):
            ConfigurationError(
                f'Decoder output({self.definition_decoder.get_output_dim()}) dim is larger than'
                f'vocabulary size({self.vocab.get_vocab_size(definition_namespace)}).')
        if self.definition_decoder.get_output_dim() < self.vocab.get_vocab_size(definition_namespace):
            self.limited_word_vocab_size = self.definition_decoder.get_output_dim()

        # self.pdist = nn.PairwiseDistance(p=2)
        self.pdist = lambda x, y: torch.mean((x-y)**2, dim=1)
        self.metrics = {'consistency_loss': EuclideanDistance()}

        initializer(self)

    def _encode_definition(self, definition: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # [batch_size, seq_len]
        definition_mask = util.get_text_field_mask(definition)
        # [batch_size, seq_len, emb_dim]
        embedded_definition = self.text_embedder(definition)

        # either [batch_size, emb_dim] or [batch_size, seq_len, emb_dim] 
        encoded_definition = self.definition_encoder(embedded_definition, definition_mask)
        # if len(encoded_definition.size()) == 3:
        if self.definition_pooling == 'last':
            # [batch_size, emb_dim]
            encoded_definition = util.get_final_encoder_states(encoded_definition, definition_mask)
        elif self.definition_pooling == 'max':
            # encoded_definition = F.adaptive_max_pool1d(encoded_definition.transpose(1, 2), 1).squeeze(2)
            encoded_definition = util.masked_max(encoded_definition, definition_mask.unsqueeze(2), dim=1)
        elif self.definition_pooling == 'mean':
            # encoded_definition = F.adaptive_avg_pool1d(encoded_definition.transpose(1, 2), 1).squeeze(2)
            encoded_definition = util.masked_mean(encoded_definition, definition_mask.unsqueeze(2), dim=1)
        elif self.definition_pooling == 'self-attentive':
            self_attentive_logits = self.self_attentive_pooling_projection(encoded_definition).squeeze(2)
            self_weights = util.masked_softmax(self_attentive_logits, definition_mask)
            encoded_definition = util.weighted_sum(encoded_definition, self_weights)
        # [batch_size, emb_dim]
        definition_embedding = self.definition_feedforward(encoded_definition)

        # [batch_size, vocab_size(num_class)]
        definition_logits = self.definition_decoder(definition_embedding)
        # [batch_size, seq_len, vocab_size]
        sequence_definition_logits = definition_logits.unsqueeze(1).repeat(1, definition_mask.size(1), 1)

        # ``average`` can be None, "batch", or "token" 
        # loss for ``average==None`` is a vector of shape (batch_size,); otherwise, a scalar
        targets = definition['tokens'].clone()
        if self.limited_word_vocab_size is not None:
            targets[targets >= self.limited_word_vocab_size] = self._oov_index
        cross_entropy_loss = util.sequence_cross_entropy_with_logits(
            sequence_definition_logits,
            targets,
            # definition['tokens'],
            weights=definition_mask,
            average='token'
        )

        return {
            "definition_embedding": definition_embedding,
            "cross_entropy_loss": cross_entropy_loss
        }

    @overrides
    def forward(
        self,
        definition: Dict[str, torch.LongTensor],
        word: Dict[str, torch.LongTensor] = None,
        word_to_definition: torch.Tensor = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:

        output_dict = {}
        output_dict.update(self._encode_definition(definition))
        output_dict['loss'] = self.alpha * output_dict['cross_entropy_loss']

        if self.beta > 0 and word is not None:
            # [batch_size, seq_len(1)]
            word_in_definition_mask = (word_to_definition != self._oov_index).float()
            # [batch_size]
            word_in_definition_mask = word_in_definition_mask.squeeze(dim=1)

            # [batch_size, seq_len(1), emb_dim]
            embedded_word = self.text_embedder({'tokens': word_to_definition})
            # [batch_size, emb_dim]
            embedded_word = embedded_word.squeeze(dim=1)

            mse = self.pdist(output_dict['definition_embedding'], embedded_word)
            consistency_loss = util.masked_mean(mse, word_in_definition_mask, dim=0)
            output_dict['consistency_loss'] = consistency_loss

            output_dict['loss'] += self.beta * output_dict['consistency_loss']

            for metric in self.metrics.values():
                metric(output_dict['definition_embedding'], embedded_word, word_in_definition_mask)

        return output_dict

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        # [batch_size, emb_dim]
        definition_embedding = output_dict['definition_embedding'].cpu().data.numpy()
        embedding_matrix = self.text_embedder._token_embedders['tokens'].weight.cpu().data.numpy()

        # GPU version
        # gpu_res = faiss.StandardGpuResources()
        # faiss_index = faiss.GpuIndexFlatL2(gpu_res, 300)
        faiss_index = faiss.IndexFlatL2(self.text_embedder.get_output_dim()) # CPU version
        faiss_index.add(embedding_matrix)

        distances, indexes = faiss_index.search(definition_embedding, 100)
        words = [
            self.vocab.get_token_from_index(x, namespace=self.definition_namespace)
            for x in indexes[:, 0]
        ]
        top10_words = [
            [self.vocab.get_token_from_index(x, namespace=self.definition_namespace) for x in batch]
            for batch in indexes[:, :10]
        ]
        top100_words = [
            [self.vocab.get_token_from_index(x, namespace=self.definition_namespace) for x in batch]
            for batch in indexes[:, :100]
        ]

        output_dict['word'] = words
        output_dict['top10_words'] = top10_words
        output_dict['top100_words'] = top100_words
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {
            metric_name: metric.get_metric(reset)
            for metric_name, metric in self.metrics.items()
        }
        return metrics_to_return
