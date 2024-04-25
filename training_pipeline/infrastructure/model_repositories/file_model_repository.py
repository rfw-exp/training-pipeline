import pathlib

from xgboost import XGBClassifier

from training_pipeline.domain.interfaces.model_repository import (
    ModelRepository,
)


class FileModelRepository(ModelRepository):
    def __init__(self, base_path: str, filetype: str = "json"):
        self._base_path = base_path
        self._filetype = filetype

    def save_model(self, identity: str, model: XGBClassifier):
        path = pathlib.Path(self._base_path).joinpath(
            f"{identity}.{self._filetype}"
        )
        model.save_model(path.as_posix())

    def load_model(self, identity: str) -> XGBClassifier:
        pass
