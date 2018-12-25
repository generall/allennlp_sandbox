from allennlp.common import JsonDict
from allennlp.data import Instance
from allennlp.models import load_archive
from allennlp.predictors import Predictor
from overrides import overrides


@Predictor.register('oracle')
class PaperClassifierPredictor(Predictor):
    """Predictor wrapper for the AcademicPaperClassifier"""

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        text = json_dict['text']
        instance = self._dataset_reader.text_to_instance(text)

        return instance


if __name__ == '__main__':
    question = {
        "text": "ykm god icy wda ans2 rlt rlt mbi icy dyz nvc"
    }

    model_path = 'data/stats/model.tar.gz'

    archive = load_archive(model_path)
    predictor = Predictor.from_archive(archive, 'oracle')

    result = predictor.predict_json(question)

    print(result)
