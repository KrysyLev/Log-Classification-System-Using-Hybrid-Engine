import joblib
from sentence_transformers import SentenceTransformer


class BertClassifier:
    """
    BertClassifier provides a classification interface for log messages using a 
    SentenceTransformer for embedding and a pre-trained classifier for prediction.
    Attributes:
        transformer_model (SentenceTransformer): The sentence transformer model used to encode log messages.
        model_classification (sklearn.base.BaseEstimator): The classification model loaded from a joblib file.
        threshold (float): The probability threshold below which a log message is considered 'Unclassified'.
    Args:
        model_path (str): Path to the serialized classification model. Default is "models/logistic_classifier.joblib".
        transformer_name (str): Name or path of the sentence transformer model. Default is "all-MiniLM-L6-v2".
        threshold (float): Probability threshold for classification. Default is 0.52.
    Methods:
        classify(log_message):
            Encodes the input log message, predicts its class probabilities, and returns the predicted label.
            If the highest probability is below the threshold, returns "Unclassified".
    Example:
        >>> classifier = BertClassifier()
        >>> label = classifier.classify("Error: Disk quota exceeded")
        >>> print(label)
        "Error"
    """
    def __init__(
        self,
        model_path="models/logistic_classifier.joblib",
        transformer_name="all-MiniLM-L6-v2",
        threshold=0.52,
    ):
        self.transformer_model = SentenceTransformer(transformer_name)
        self.model_classification = joblib.load(model_path)
        self.threshold = threshold

    def classify(self, log_message):
        embeddings = self.transformer_model.encode([log_message])
        probabilities = self.model_classification.predict_proba(embeddings)[0]
        if max(probabilities) < self.threshold:
            return "Unclassified"
        predicted_label = self.model_classification.predict(embeddings)[0]
        return predicted_label


if __name__ == "__main__":
    classifier = BertClassifier()
    logs = [
        "alpha.osapi_compute.wsgi.server - 12.10.11.1 - API returned 404 not found error",
        "GET /v2/3454/servers/detail HTTP/1.1 RCODE   404 len: 1583 time: 0.1878400",
        "System crashed due to drivers errors when restarting the server",
        "Hey bro, chill ya!",
        "Multiple login failures occurred on user 6454 account",
        "Server A790 was restarted unexpectedly during the process of data transfer"
    ]
    for log in logs:
        label = classifier.classify(log)
        print(log, "->", label)
