from training_pipeline.application.training_controller import (
    TrainingController,
)


def train_model():
    controller = TrainingController()
    controller.train()


if __name__ == "__main__":
    train_model()
