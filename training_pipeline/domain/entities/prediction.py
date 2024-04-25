import pandera as pa
from pandera.typing import Index, Series


class PredictionSchema(pa.DataFrameModel):
    id: Index[str]
    fraud_flag: Series[bool]
