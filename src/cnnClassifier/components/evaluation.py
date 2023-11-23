import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_dataclass import EvaluationConfig
from cnnClassifier.utils.common import save_json

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        # Constructor to initialize the Evaluation object with a given configuration.
        self.config = config
    
    def _valid_generator(self):
        # Method to set up a data generator for validation.

        # Data augmentation and scaling configurations for validation data.
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
        )

        # Data flow configurations for image preprocessing.
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # Create a data generator for validation data.
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        # Set up a generator for validation data flow from the specified directory.
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        # Static method to load a trained model from a specified path.
        return tf.keras.models.load_model(path)

    def evaluation(self):
        # Method to perform evaluation using a trained model on the validation data.

        # Load the trained model.
        self.model = self.load_model(self.config.path_of_model)

        # Set up the validation data generator.
        self._valid_generator()

        # Perform evaluation on the validation data and store the results in 'self.score'.
        self.score = self.model.evaluate(self.valid_generator)

    def save_score(self):
        # Method to save the evaluation scores to a JSON file.

        # Create a dictionary with loss and accuracy scores.
        scores = {"loss": self.score[0], "accuracy": self.score[1]}

        # Save the scores to a JSON file.
        save_json(path=Path("scores.json"), data=scores)
