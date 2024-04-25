from abc import ABC, abstractmethod

from xgboost import XGBClassifier


class ModelRepository(ABC):
    @abstractmethod
    def save_model(self, identity: str, model: XGBClassifier) -> None:
        pass

    @abstractmethod
    def load_model(self, identity: str) -> XGBClassifier:
        pass
