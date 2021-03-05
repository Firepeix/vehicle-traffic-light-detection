import tensorflow
from tensorflow import keras
import tensorflow_datasets
from utils import *
from helpers import *
import os
class Trainer:
    def __init__(self, dataset_dir, model_dir, classes_quantity):
        self.__model_dir = model_dir
        self.__dataset_dir = dataset_dir
        self.__model = None
        self.__classes_quantity = classes_quantity
        self.__dataset_information = None
        self.__train_dataset = None
        self.__validation_dataset = None
        self.__autotune = tensorflow.data.experimental.AUTOTUNE
        self.batch_size = 2
        self.__label_encoder = LabelEncoder()
        self.__set_learning()
        self.__set_callbacks()

    def __set_learning(self):
        learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
        learning_rate_boundaries = [125, 250, 500, 240000, 360000]
        self.__learning_rate_callback = tensorflow.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=learning_rate_boundaries, values=learning_rates
        )

    def __set_callbacks(self):
        self.__callbacks_list = [
            tensorflow.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.__model_dir, "weights" + "_epoch_{epoch}"),
                monitor="loss",
                save_best_only=False,
                save_weights_only=True,
                verbose=1,
            )
        ]

    def create_model(self, backbone):
        self.__model = RetinaNet(self.__classes_quantity, backbone)

    def compile(self):
        loss_callback = RetinaNetLoss(self.__classes_quantity)
        optimizer = tensorflow.optimizers.SGD(learning_rate=self.__learning_rate_callback, momentum=0.9)
        self.__model.compile(loss=loss_callback, optimizer=optimizer)

    def load_dataset(self, dataset_path):
        split = ["train", "validation"]
        data_dir = self.__dataset_dir
        (t_dt, val_dt), dt_info = tensorflow_datasets.load(dataset_path, split=split, with_info=True, data_dir=data_dir)
        self.__dataset_information = dt_info
        self.__train_dataset = t_dt
        self.__validation_dataset = val_dt

    def prepare_train_dataset(self):
        self.__train_dataset = self.__train_dataset.map(preprocess_data, num_parallel_calls=self.__autotune)
        self.__train_dataset = self.__train_dataset.shuffle(8 * self.batch_size)
        self.__train_dataset = self.__train_dataset.padded_batch(
            batch_size=self.batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
        )
        self.__train_dataset = self.__train_dataset.map(
            self.__label_encoder.encode_batch, num_parallel_calls=self.__autotune
        )
        self.__train_dataset = self.__train_dataset.apply(tf.data.experimental.ignore_errors())
        self.__train_dataset = self.__train_dataset.prefetch(self.__autotune)

    def prepare_validation_dataset(self):
        self.__validation_dataset = self.__validation_dataset.map(preprocess_data, num_parallel_calls=self.__autotune)
        self.__validation_dataset = self.__validation_dataset.padded_batch(
            batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True
        )
        self.__validation_dataset = self.__validation_dataset.map(self.__label_encoder.encode_batch, num_parallel_calls=self.__autotune)
        self.__validation_dataset = self.__validation_dataset.apply(tf.data.experimental.ignore_errors())
        self.__validation_dataset = self.__validation_dataset.prefetch(self.__autotune)

    def fit(self):
        # Uncomment the following lines, when training on full dataset
        # train_steps_per_epoch = dataset_info.splits["train"].num_examples // batch_size
        # val_steps_per_epoch = \
        #     dataset_info.splits["validation"].num_examples // batch_size

        # train_steps = 4 * 100000
        # epochs = train_steps // train_steps_per_epoch

        epochs = 1

        self.__model.fit(
            self.__train_dataset.take(100),
            validation_data=self.__validation_dataset.take(50),
            epochs=epochs,
            callbacks=self.__callbacks_list,
            verbose=1,
        )

class Detector:
    def __init__(self, model_dir, classes_quantity):
        self.__model_dir = model_dir
        self.__inference_model = None
        self.__model = None
        self.__classes_quantity = classes_quantity
        self.__dataset_information = None

    def load_dataset(self, data_dir, dataset_path):
        (train_dataset, val_dataset), dataset_info = tensorflow_datasets.load(dataset_path, split=["train", "validation"], with_info=True, data_dir=data_dir)
        self.__dataset_information = dataset_info

    def create_model(self, backbone):
        self.__model = RetinaNet(self.__classes_quantity, backbone)

    def load_weights(self):
        self.__model.load_weights(tensorflow.train.latest_checkpoint(self.__model_dir))

    def create_inference_model(self):
        image = tensorflow.keras.Input(shape=[None, None, 3], name="image")
        predictions = self.__model(image, training=False)
        detections = DecodePredictions(confidence_threshold=0.26)(image, predictions)
        self.__inference_model = keras.Model(inputs=image, outputs=detections)

    def predict(self, image):
        return self.__inference_model.predict(image)

    def get_class_names(self, detections):
        num_detections = detections.valid_detections[0]
        int2str = self.__dataset_information.features["objects"]["label"].int2str
        return [
            int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]
        ]