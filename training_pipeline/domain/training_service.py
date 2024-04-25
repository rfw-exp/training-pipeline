from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from training_pipeline.domain.entities.consumer import ConsumerSchema
from training_pipeline.domain.interfaces.data_reader import DataReader
from training_pipeline.domain.interfaces.model_repository import (
    ModelRepository,
)
from training_pipeline.domain.model_evaluator import ModelEvaluator
from training_pipeline.domain.model_trainer import ModelTrainer


class TrainingService:
    def __init__(
        self,
        data_reader: DataReader,
        model_repository: ModelRepository,
    ):
        self._data_reader = data_reader
        self._model_repository = model_repository

    def train_model(self) -> XGBClassifier:
        data = self._data_reader.read()

        features = list(data.columns)
        features.remove(ConsumerSchema.fraud_flag)
        X = data[features]
        y = data.fraud_flag
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        training_data = X_train.join(y_train, how="inner")
        model = ModelTrainer().train_model(training_data)

        predictions = model.predict(X_test)

        evaluation = ModelEvaluator().evaluate_model(y_test, predictions)

        print(evaluation)

        self._model_repository.save_model(identity="xgboost", model=model)

        return model
