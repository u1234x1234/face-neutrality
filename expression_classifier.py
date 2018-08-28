import numpy as np
import mxnet as mx


class NNExpressionClassifier:
    def __init__(self, model_prefix: str):
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 0)
        self.model = mx.model.FeedForward(sym, arg_params=arg_params, aux_params=aux_params, numpy_batch_size=1)

    def predict(self, mouth_image: np.ndarray) -> (float, float):
        """Predict probabilities of the two expressions: `smile`, `mouth_open`
        """
        data = mouth_image.transpose(2, 0, 1)[np.newaxis, :, :, :]
        smile_prob, open_mouth_prob = self.model.predict(data)[0]
        return smile_prob, open_mouth_prob
