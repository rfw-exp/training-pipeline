from pydantic import BaseModel


class ModelEvaluation(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
