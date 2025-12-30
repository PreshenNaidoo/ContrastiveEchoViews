import keras_cv
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers

from augmentation_util import *

# Hyperparameters for the algorithm
NETWORK_WIDTH = 128
TEMPERATURE = 0.1
LARGE_NUM = 1e9

IMAGE_SIZE = (224, 224, 3)

# Augmentations for contrastive and supervised training
AUG_CONTRASTIVE = {
    "image_size": IMAGE_SIZE[0],
    "image_channels": IMAGE_SIZE[-1],
    "min_area": 0.8, #0.6,#0.25,
    "brightness": 0.8, #0.6,
    "jitter": 0.2
}
AUG_SEGMENTATION = {
    "image_size": IMAGE_SIZE[0],
    "image_channels": IMAGE_SIZE[-1],
    "min_area": 0.75,
    "brightness": 0.3,
    "jitter": 0.1
}

# Define the contrastive model with model-subclassing
class SimCLR_Contrastive_Model(keras.Model):
    def __init__(self, encoder):
        super().__init__()

        self.temperature = TEMPERATURE
        self.contrastive_augmenter = get_augmenter(**AUG_CONTRASTIVE)
        self.classification_augmenter = get_augmenter(**AUG_SEGMENTATION)
        self.encoder = encoder
        # Non-linear MLP as projection head
        #Get output shape of encoder
        encoder_output_shape = self.encoder.output_shape
        encoder_output_shape = (encoder_output_shape[1], encoder_output_shape[2], encoder_output_shape[3])

        self.projection_head = keras.Sequential(
        [
                keras.Input(shape=encoder_output_shape),  # output shape of the encoder
                layers.Flatten(),
                layers.Dense(NETWORK_WIDTH, activation="relu"),
                layers.Dense(NETWORK_WIDTH),
            ],
            name="Projection_Head",
        )

        self.encoder.summary()
        self.projection_head.summary()

    def compile(self, contrastive_optimizer, **kwargs):
        super().compile(**kwargs)

        self.contrastive_optimizer = contrastive_optimizer


        self.contrastive_loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [
            self.contrastive_loss_tracker,
        ]

    #     # InfoNCE loss (information noise-contrastive estimation)
    #     # NT-Xent loss (normalized temperature-scaled cross entropy)
    #
    #     # Cosine similarity: the dot product of the l2-normalized feature vectors
    #
    #
    #
    #     return loss

    def contrastive_loss(self, projections_1, projections_2):
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)

        # Cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        similarities = (
            tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature
        )

        # The similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)

        # The temperature-scaled similarities are used as logits for cross-entropy
        # a symmetrized version of the loss is used here
        loss_1_2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True
        )
        return (loss_1_2 + loss_2_1) / 2

    def train_step(self, data):
        images = data #first axis is the batch

        # Each image is augmented twice, differently
        augmented_images_1 = self.contrastive_augmenter(images, training=True)
        augmented_images_2 = self.contrastive_augmenter(images, training=True)

        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1, training=True)
            features_2 = self.encoder(augmented_images_2, training=True)
            # The representations are passed through a projection mlp
            projections_1 = self.projection_head(features_1, training=True)
            projections_2 = self.projection_head(features_2, training=True)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)

        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        self.contrastive_loss_tracker.update_state(contrastive_loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images = data  # first axis is the batch

        # Each image is augmented twice, differently
        augmented_images_1 = self.contrastive_augmenter(images, training=False)
        augmented_images_2 = self.contrastive_augmenter(images, training=False)

        features_1 = self.encoder(augmented_images_1, training=False)
        features_2 = self.encoder(augmented_images_2, training=False)
        # The representations are passed through a projection mlp
        projections_1 = self.projection_head(features_1, training=False)
        projections_2 = self.projection_head(features_2, training=False)
        contrastive_loss = self.contrastive_loss(projections_1, projections_2)

        self.contrastive_loss_tracker.update_state(contrastive_loss)

        return {m.name: m.result() for m in self.metrics}

    def save(self, filepath, overwrite=True, save_format=None, **kwargs):
        print(f'\n **SAVED MODEL: f{filepath} \n')
        self.encoder.save(filepath, overwrite, save_format)
