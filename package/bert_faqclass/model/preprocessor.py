import tensorflow_hub as hub


class Preprocessor:
    def __init__(self, preprocessor_url: str):
        self._preprocessor = hub.KerasLayer(preprocessor_url)

    def preprocess(self, data: [str]):
        return self._preprocessor(data)
