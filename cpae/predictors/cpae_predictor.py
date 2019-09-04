from typing import List
from overrides import overrides


from allennlp.data import Instance
from allennlp.common.util import JsonDict
from allennlp.predictors import Predictor


@Predictor.register('cpae_definition_embedding_generator')
class CPAEDefinitionEmbeddingGenerator(Predictor):

    @overrides
    def predict_json(self, inputs: JsonDict) -> str:
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)

        return '{} {}'.format(
            inputs['word'],
            ' '.join(str(val) for val in output_dict['definition_embedding'])
        )

    @overrides
    def predict_batch_json(self, inputs: List[JsonDict]) -> List[str]:
        instances = self._batch_json_to_instances(inputs)
        output_dicts = self.predict_batch_instance(instances)

        results = []
        for inp, od in zip(inputs, output_dicts):
            results.append('{} {}'.format(
                inp['word'],
                ' '.join(str(val) for val in od['definition_embedding'])
            ))
        return results

    @overrides
    def _batch_json_to_instances(self, json_dicts: List[JsonDict]) -> List[Instance]:
        instances = []
        for json_dict in json_dicts:
            instances.append(self._json_to_instance(json_dict))

        return instances

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        definition = json_dict['definition']
        # word = json_dict['word']
        return self._dataset_reader.text_to_instance(definition=definition, word=None)
