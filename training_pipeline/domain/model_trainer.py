from pandera.typing import DataFrame
from xgboost import XGBClassifier

from training_pipeline.domain.entities.consumer import ConsumerSchema


class ModelTrainer:
    def train_model(
        self, training_data: DataFrame[ConsumerSchema]
    ) -> XGBClassifier:
        X_train = training_data.drop(columns=[ConsumerSchema.fraud_flag])
        y_train = training_data[ConsumerSchema.fraud_flag]
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        model.fit(X_train, y_train)
        return model
