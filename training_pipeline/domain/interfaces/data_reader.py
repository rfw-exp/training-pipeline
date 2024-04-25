from abc import ABC, abstractmethod

from pandera.typing import DataFrame

from training_pipeline.domain.entities.consumer import ConsumerSchema


class DataReader(ABC):
    @abstractmethod
    def read(self) -> DataFrame[ConsumerSchema]:
        pass
