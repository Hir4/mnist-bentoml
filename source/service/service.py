import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

BENTO_MODEL_TAG = "sklearn_model:as2jxmvifkqvhw7m"

classifier_runner = bentoml.sklearn.get(BENTO_MODEL_TAG).to_runner()

mnist_service = bentoml.Service("mnist_service", runners=[classifier_runner])

@mnist_service.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_data):
  return classifier_runner.predict.run(input_data)