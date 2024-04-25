import pandas as pd
from pandera.typing import DataFrame

from training_pipeline.domain.entities.consumer import ConsumerSchema
from training_pipeline.domain.interfaces.data_reader import DataReader


class FileDataReader(DataReader):
    def __init__(self, path: str):
        self._path = path

    def read(self) -> DataFrame[ConsumerSchema]:
        df = DataFrame[ConsumerSchema](
            pd.read_csv(self._path, index_col=ConsumerSchema.id)
        )
        return df
