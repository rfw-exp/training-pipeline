from pandera.typing import DataFrame
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from training_pipeline.domain.entities.model_evaluation import ModelEvaluation
from training_pipeline.domain.entities.prediction import PredictionSchema


class ModelEvaluator:
    def evaluate_model(
        self,
        predictions: DataFrame[PredictionSchema],
        truth: DataFrame[PredictionSchema],
    ):
        accuracy = accuracy_score(truth, predictions)
        precision = precision_score(truth, predictions)
        recall = recall_score(truth, predictions)
        f1 = f1_score(truth, predictions)
        return ModelEvaluation(
            accuracy=accuracy, precision=precision, recall=recall, f1=f1
        )
