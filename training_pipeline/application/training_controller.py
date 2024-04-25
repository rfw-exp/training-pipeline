from xgboost import XGBClassifier

from training_pipeline.domain.model_trainer import ModelTrainer
from training_pipeline.domain.training_service import TrainingService
from training_pipeline.infrastructure.data_readers.file_data_reader import (
    FileDataReader,
)
from training_pipeline.infrastructure.model_repositories.file_model_repository import (
    FileModelRepository,
)


class TrainingController:
    def __init__(self):
        data_reader = FileDataReader(path="data/consumers.csv")
        model_repository = FileModelRepository(base_path="data/models/")
        self._service = TrainingService(
            data_reader=data_reader,
            model_repository=model_repository,
        )

    def train(self) -> XGBClassifier:
        model = self._service.train_model()
        return model
