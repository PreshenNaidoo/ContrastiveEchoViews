"""Evaluation utilities for SSL video classification experiments."""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import random
import math
import cv2
import itertools
import time
import tensorflow_addons as tfa
import pandas as pd

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import architecture_ResNet
import utils
from vc_model_simclr import *
from vc_model_btwin import *

from architecture_ResNet import *

IMAGE_SIZE = (224, 224, 3)
BATCH_SIZE = 14#24 # 1 for video at a time
BACKBONES = ['Resnet50', 'Xception', 'DenseNet121']
NUM_CLASSES = 17  # 53
MIN_FRAMES = 40
SKIP_FRAMES = 8 # =4 means select every 4th frame in a video
TOTAL_FRAMES = MIN_FRAMES//SKIP_FRAMES

AUG_MIN_AREA = 0.8
AUG_ROTATION = 0.05


def plot_confusion_matrix_2(cm,
                            target_names,
                            title='Confusion matrix',
                            cmap=None,
                            normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions


    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    font_size = 18
    label_size = 20

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(18, 15))  # 8, 6
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=label_size)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=label_size)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90, fontsize=font_size)
        plt.yticks(tick_marks, target_names, fontsize=font_size)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     fontsize=font_size,
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     fontsize=font_size,
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=label_size)
    plt.xlabel('Predicted label', fontsize=label_size)
    plt.show()


class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Implements an LR scheduler that warms up the learning rate for some training steps
    (usually at the beginning of the training) and then decays it
    with CosineDecay (see https://arxiv.org/abs/1608.03983)
    """

    def __init__(
            self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super(WarmUpCosine, self).__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")
        learning_rate = (
                0.5
                * self.learning_rate_base
                * (
                        1
                        + tf.cos(
                    self.pi
                    * (tf.cast(step, tf.float32) - self.warmup_steps)
                    / float(self.total_steps - self.warmup_steps)
                )
                )
        )

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                            self.learning_rate_base - self.warmup_learning_rate
                    ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )


def split_data(data_folder, perc_train, perc_val_of_train, perc_test, merge=False):
    dirs = os.listdir(data_folder)

    train_files, train_labels = [], []
    val_files, val_labels = [], []
    test_files, test_labels = [], []
    class_count = {}  # num frames per class
    class_vid_count = {}  # num vids per class
    class_lookup = {}
    class_label = 0
    num_excluded = 0
    excluded_cnt = {}

    class_files_dict = {}
    for dir in dirs:

        path = os.path.join(data_folder, dir)
        files = os.listdir(path)
        files.sort()
        num_files = len(files)

        video_dict = {}
        for file in files:
            video_name = file[:file.rindex('_')]
            video_name = os.path.join(path, video_name)

            if video_name in video_dict:
                video_dict[video_name].append(os.path.join(path, file))
            else:
                video_dict[video_name] = [os.path.join(path, file)]

        class_files_dict[dir] = video_dict

    if merge:
        # merge these:
        parasternal_av_pv = {}
        plax_pv = class_files_dict['PLAX_PV']
        psax_qv = class_files_dict['PSAX_AV']
        psax_pv = class_files_dict['PSAX_PV']
        for vid_name, filepath in plax_pv.items():
            parasternal_av_pv[vid_name] = filepath
        for vid_name, filepath in psax_qv.items():
            parasternal_av_pv[vid_name] = filepath
        for vid_name, filepath in psax_pv.items():
            parasternal_av_pv[vid_name] = filepath
        del class_files_dict['PLAX_PV']
        del class_files_dict['PSAX_AV']
        del class_files_dict['PSAX_PV']
        class_files_dict['parastnl_av_pv'] = parasternal_av_pv

        # merge these:
        subcostal = {}
        subcostal_h = class_files_dict['Subcostal_heart']
        subcostal_i = class_files_dict['Subcostal_IVC']
        for vid_name, filepath in subcostal_h.items():
            subcostal[vid_name] = filepath
        for vid_name, filepath in subcostal_i.items():
            subcostal[vid_name] = filepath
        del class_files_dict['Subcostal_heart']
        del class_files_dict['Subcostal_IVC']
        class_files_dict['subcostal'] = subcostal

    for class_name, videos_dict in class_files_dict.items():
        num_vids_all = len(videos_dict)

        vids_list = []
        vids_list_all = list(videos_dict.keys())
        cnt_excl = 0
        for vid in vids_list_all:
            frames = videos_dict[vid]
            if len(frames) >= MIN_FRAMES:
                vids_list.append(vid)
            else:
                cnt_excl += 1

        num_vids = len(vids_list)
        class_vid_count[class_name] = num_vids

        num_vids_train = int(num_vids * perc_train)
        num_vids_val = int(num_vids_train * perc_val_of_train)  # val is a perc of train (this wwas done in the original paper)
        num_vids_train = num_vids_train - num_vids_val
        num_vids_test = int(num_vids * perc_test)

        random.shuffle(vids_list)
        test_vids = vids_list[0: num_vids_test]
        train_vids = vids_list[num_vids_test: num_vids_test + num_vids_train]
        val_vids = vids_list[num_vids_test + num_vids_train:]

        labels_train = [class_label] * len(train_vids)
        labels_val = [class_label] * len(val_vids)
        labels_test = [class_label] * len(test_vids)

        train_files.extend(train_vids)
        val_files.extend(val_vids)
        test_files.extend(test_vids)

        train_labels.extend(labels_train)
        val_labels.extend(labels_val)
        test_labels.extend(labels_test)

        excluded_cnt[class_name] = cnt_excl
        class_count[class_name] = len(train_vids) + len(val_vids) + len(test_vids)
        class_lookup[class_label] = class_name
        class_label += 1

    temp = list(zip(train_files, train_labels))
    random.shuffle(temp)
    train_files, train_labels = zip(*temp)

    temp = list(zip(val_files, val_labels))
    random.shuffle(temp)
    val_files, val_labels = zip(*temp)

    temp = list(zip(test_files, test_labels))
    random.shuffle(temp)
    test_files, test_labels = zip(*temp)

    # uncomment to make it an exact multiple of batch size
    per_step = len(train_files) / float(BATCH_SIZE)
    num_train = math.floor(per_step) * BATCH_SIZE
    train_files = train_files[:num_train]
    train_labels = train_labels[:num_train]

    per_step = len(val_files) / float(BATCH_SIZE)
    num_val = math.floor(per_step) * BATCH_SIZE
    val_files = val_files[:num_val]
    val_labels = val_labels[:num_val]


    print(f'NUM EXCLUDED: {num_excluded}')

    return (list(train_files), list(train_labels),
            list(val_files), list(val_labels),
            list(test_files), list(test_labels),
            class_count, class_lookup, class_vid_count, excluded_cnt)

def split_data_temp(data_folder, train_txt, val_txt, test_txt, merge=False):
    train_dict, val_dict, test_dict = {}, {}, {}
    train_list, val_list, test_list=[],[],[]
    with open(train_txt) as file:
        for line in file:
            frame = line.rstrip()
            video_name = frame[:frame.rindex('_')]
            if video_name not in train_dict:
                train_dict[video_name] = 1
            train_list.append(frame)
    with open(val_txt) as file:
        for line in file:
            frame = line.rstrip()
            video_name = frame[:frame.rindex('_')]
            if video_name not in val_dict:
                val_dict[video_name] = 1
            val_list.append(frame)
    with open(test_txt) as file:
        for line in file:
            frame = line.rstrip()
            video_name = frame[:frame.rindex('_')]
            if video_name not in test_dict:
                test_dict[video_name] = 1
            test_list.append(frame)

    dirs = os.listdir(data_folder)

    train_files, train_labels = [], []
    val_files, val_labels = [], []
    test_files, test_labels = [], []
    class_count = {}  # num frames per class
    class_vid_count = {}  # num vids per class
    class_lookup = {}
    class_label = 0
    num_excluded = 0
    excluded_cnt = {}

    class_files_dict = {}
    for dir in dirs:

        path = os.path.join(data_folder, dir)
        files = os.listdir(path)
        files.sort()
        num_files = len(files)

        video_dict = {}
        for file in files:
            video_name = file[:file.rindex('_')]
            video_name = os.path.join(path, video_name)

            if video_name in video_dict:
                video_dict[video_name].append(os.path.join(path, file))
            else:
                video_dict[video_name] = [os.path.join(path, file)]

        class_files_dict[dir] = video_dict

    if merge:
        # merge these:
        parasternal_av_pv = {}
        plax_pv = class_files_dict['PLAX_PV']
        psax_qv = class_files_dict['PSAX_AV']
        psax_pv = class_files_dict['PSAX_PV']
        for vid_name, filepath in plax_pv.items():
            parasternal_av_pv[vid_name] = filepath
        for vid_name, filepath in psax_qv.items():
            parasternal_av_pv[vid_name] = filepath
        for vid_name, filepath in psax_pv.items():
            parasternal_av_pv[vid_name] = filepath
        del class_files_dict['PLAX_PV']
        del class_files_dict['PSAX_AV']
        del class_files_dict['PSAX_PV']
        class_files_dict['parastnl_av_pv'] = parasternal_av_pv

        # merge these:
        subcostal = {}
        subcostal_h = class_files_dict['Subcostal_heart']
        subcostal_i = class_files_dict['Subcostal_IVC']
        for vid_name, filepath in subcostal_h.items():
            subcostal[vid_name] = filepath
        for vid_name, filepath in subcostal_i.items():
            subcostal[vid_name] = filepath
        del class_files_dict['Subcostal_heart']
        del class_files_dict['Subcostal_IVC']
        class_files_dict['subcostal'] = subcostal

    for class_name, videos_dict in class_files_dict.items():
        num_vids_all = len(videos_dict)

        vids_list = []
        vids_list_all = list(videos_dict.keys())
        cnt_excl = 0
        for vid in vids_list_all:
            frames = videos_dict[vid]
            if len(frames) >= MIN_FRAMES:
                vids_list.append(vid)
            else:
                cnt_excl += 1

        num_vids = len(vids_list)
        class_vid_count[class_name] = num_vids

        train_vids, val_vids, test_vids=[],[],[]
        for vid in vids_list:
            if vid in train_dict:
                train_vids.append(vid)
            elif vid in val_dict:
                val_vids.append(vid)
            elif vid in test_dict:
                test_vids.append(vid)

        labels_train = [class_label] * len(train_vids)
        labels_val = [class_label] * len(val_vids)
        labels_test = [class_label] * len(test_vids)

        train_files.extend(train_vids)
        val_files.extend(val_vids)
        test_files.extend(test_vids)

        train_labels.extend(labels_train)
        val_labels.extend(labels_val)
        test_labels.extend(labels_test)

        excluded_cnt[class_name] = cnt_excl
        class_count[class_name] = len(train_vids) + len(val_vids) + len(test_vids)
        class_lookup[class_label] = class_name
        class_label += 1

    temp = list(zip(train_files, train_labels))
    random.shuffle(temp)
    train_files, train_labels = zip(*temp)

    temp = list(zip(val_files, val_labels))
    random.shuffle(temp)
    val_files, val_labels = zip(*temp)

    temp = list(zip(test_files, test_labels))
    random.shuffle(temp)
    test_files, test_labels = zip(*temp)

    # uncomment to make it an exact multiple of batch size
    per_step = len(train_files) / float(BATCH_SIZE)
    num_train = math.floor(per_step) * BATCH_SIZE
    train_files = train_files[:num_train]
    train_labels = train_labels[:num_train]
    #
    per_step = len(val_files) / float(BATCH_SIZE)
    num_val = math.floor(per_step) * BATCH_SIZE
    val_files = val_files[:num_val]
    val_labels = val_labels[:num_val]


    print(f'NUM EXCLUDED: {num_excluded}')

    return (list(train_files), list(train_labels),
            list(val_files), list(val_labels),
            list(test_files), list(test_labels),
            class_count, class_lookup, class_vid_count, excluded_cnt)


def process_video_only(video_file):
    frames = []
    for i in range(0, MIN_FRAMES, SKIP_FRAMES):
        frame = tf.strings.as_string(f'_{str(i).zfill(4)}.png')
        image_file = tf.strings.join([video_file, frame])
        img = tf.io.read_file(image_file)
        image = tf.image.decode_png(img, channels=3)


        h, w = image.shape[:2]

        if h != IMAGE_SIZE[0] and w != IMAGE_SIZE[1]:
            image = tf.image.resize(image, (IMAGE_SIZE[0], IMAGE_SIZE[1]))

        frames.append(image)

    frames_tensor = tf.stack(frames, 0)

    return frames_tensor


def process_video_and_label(video_file, label):
    return process_video_only(video_file), label


def get_augmenter_echo(image_size, min_area, rotation):
    zoom_factor = 1.0 - math.sqrt(min_area)
    return keras.Sequential(
        [
            keras.Input(shape=image_size),
            tf.keras.layers.RandomTranslation(zoom_factor / 2, zoom_factor / 2),
            tf.keras.layers.RandomZoom((-zoom_factor, 0.0), (-zoom_factor, 0.0)),
            tf.keras.layers.RandomRotation(rotation)
        ]
    )


def get_tf_datasets(train_files, train_labels,
                    val_files, val_labels,
                    test_files, test_labels, augmentation=False):
    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    train_ds = train_ds.map(process_video_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=BATCH_SIZE * 10).batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    val_ds = val_ds.map(process_video_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.shuffle(buffer_size=BATCH_SIZE * 10).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((test_files, test_labels))
    test_ds = test_ds.map(process_video_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    test_images_ds = tf.data.Dataset.from_tensor_slices((test_files))
    test_images_ds = test_images_ds.map(process_video_only, num_parallel_calls=tf.data.AUTOTUNE)
    test_images_ds = test_images_ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, test_images_ds


def clean_files(files_list):
    '''
    A few image files (around 8 or so) were corrupted, but it is hard to find out which ones when it fails
    inside a tensorflow mapped function. Run this once to clean the dataset and fix this problem.
    '''
    cnt_removed = 0
    for file in files_list:
        try:
            img = tf.io.read_file(file)
            img1 = tf.image.decode_png(img, channels=3)
        except:
            os.remove(file)
            cnt_removed += 1
            print(file)

    print(f'DELETED: {cnt_removed}')

class VideoAug(tf.keras.layers.Layer):
    def __init__(self, numframes, height, width, num_channels, min_area, rotation, name=None, **kwargs):
        super().__init__()
        self.numframes=numframes
        self.height = height
        self.width = width
        self.num_channels = num_channels
        self.min_area = min_area
        self.rotation = rotation
        self.zoom_factor = 1.0 - math.sqrt(min_area)
        self.aug_layer = tf.keras.Sequential(
            [
                tf.keras.layers.RandomTranslation(self.zoom_factor / 2, self.zoom_factor / 2),
                tf.keras.layers.RandomZoom((-self.zoom_factor, 0.0), (-self.zoom_factor, 0.0)),
                tf.keras.layers.RandomRotation(rotation)
            ]
        )

    def get_config(self):
        config = super(VideoAug, self).get_config()
        config.update({"numframes": self.numframes,
                       "height": self.height,
                       "width": self.width,
                       "num_channels": self.num_channels,
                       "min_area": self.min_area,
                       "rotation": self.rotation
                       })
        return config

    @tf.function
    def call(self, video):
        """
          Use the einops library to resize the tensor.

          Args:
            video: Tensor representation of the video, in the form of a set of frames.

          Return:
            A downsampled size of the video according to the new height and width it should be resized to.
        """
        # b stands for batch size, t stands for time, h stands for height,
        # w stands for width, and c stands for the number of channels.

        # input dimension:
        BATCH, TIME, WIDTH, HEIGHT, CHANNELS = video.shape#tf.shape(video)
        BATCH = BATCH_SIZE
        # move time to last
        videos = tf.transpose(video, [0, 2, 3, 4, 1])
        # combine channels and time
        channels = TIME * CHANNELS
        out_shape = (BATCH, WIDTH, HEIGHT, TIME * CHANNELS)

        videos = tf.reshape(videos, out_shape)

        augmented_data = self.aug_layer(videos)

        #reshape back to original
        augmented_data = tf.reshape(augmented_data, (BATCH, HEIGHT, WIDTH, CHANNELS, channels // CHANNELS))
        augmented_data = tf.transpose(augmented_data, [0, 4, 1, 2, 3])

        return augmented_data

@tf.function
def tf_shuffle_axis(value, axis=0, seed=None, name=None):
    perm = list(tf.range(tf.rank(value)))
    perm[axis], perm[0] = perm[0], perm[axis]
    value = tf.random.shuffle(tf.transpose(value, perm=perm))
    value = tf.transpose(value, perm=perm)
    return value

class RollingLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(RollingLayer, self).__init__()

    def build(self, input_shape):
        super(RollingLayer, self).build(input_shape)

    def call(self, inputs):
        rand = tf.random.uniform(shape=(), minval=0, maxval=MIN_FRAMES-1, dtype=tf.int32)
        rolled = tf.roll(inputs, rand, axis=1) #skip batch dim
        return rolled

class RandomDropLayer(tf.keras.layers.Layer):
    def __init__(self, rows):
        super(RandomDropLayer, self).__init__()
        self.rows = rows

    def build(self, input_shape):
        super(RandomDropLayer, self).build(input_shape)

    def call(self, inputs):
        shape = tf.shape(inputs)
        rng = tf.range(self.rows)
        rand = tf.random.shuffle(rng)
        if self.rows == 8:
            select = tf.keras.layers.Concatenate(axis=-1)([
                                                        inputs[:, rand[0]:rand[0]+1, ],
                                                        inputs[:, rand[1]:rand[1]+1, ],
                                                        inputs[:, rand[2]:rand[2]+1, ],
                                                        inputs[:, rand[3]:rand[3]+1, ]
                                                    ])
        elif self.rows == 4:
            select = tf.keras.layers.Concatenate(axis=-1)([
                inputs[:, rand[0]:rand[0] + 1, ],
                inputs[:, rand[1]:rand[1] + 1, ]
            ])
        elif self.rows == 2:
            select = tf.keras.layers.Concatenate(axis=-1)([
                inputs[:, rand[0]:rand[0] + 1, ]
            ])

        return select


def train_model(train_ds, val_ds, output_folder, run_index, num_train, model_file, linear_eval):
    model = None
    encoder = tf.keras.models.load_model(model_file, custom_objects={'WarmUpCosine': WarmUpCosine,
                                                                     'VideoAug':VideoAug,
                                                                     'RollingLayer':RollingLayer})

    #

    if linear_eval:
        encoder.trainable = False
        start_from_epoch = 30
        lr = 1e-3
    else:
        encoder.trainable = True
        start_from_epoch = 20
        lr = 1e-3/2.0

    encoder.summary()

    input = tf.keras.layers.Input(shape=(TOTAL_FRAMES, IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]))
    x = encoder(input)
    if not linear_eval:
        x = tf.keras.layers.Dense(32)(x)
    output = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = tf.keras.Model(input, output, name="classifier")


    model.summary()

    # #COSINE DECAY
    #
    #
    # #EXPONENTIAL DECAY
    #
    #
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr), #linear_eval uses default lr, fine-tuning uses 1e-4
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),#(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(2, name="top-2-accuracy"),
        ],
    )

    tf.keras.utils.plot_model(encoder, to_file=os.path.join(output_folder, f'encoder.png'),
                              show_shapes=True,
                              show_dtype=False)

    tf.keras.utils.plot_model(model, to_file=os.path.join(output_folder, f'model.png'),
                              show_shapes=True,
                              show_dtype=False)

    callbks = []
    # early stopping
    early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                     patience=10, start_from_epoch=start_from_epoch,
                                                     restore_best_weights=True)
    callbks.append(early_stopper)

    # model checkpoint to save best weights
    model_save_path = os.path.join(output_folder, f'model_run{run_index}.h5')
    model_chpt = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,
                                                    monitor='val_loss',
                                                    verbose=1,
                                                    save_weights_only=False,
                                                    save_best_only=True,
                                                    save_freq='epoch',
                                                    )
    callbks.append(model_chpt)

    # model checkpoint to save best weights
    model_save_path1 = os.path.join(output_folder, f'model_run{run_index}_val_acc.h5')
    model_chpt1 = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path1,
                                                    monitor='val_accuracy',
                                                    verbose=1,
                                                    save_weights_only=False,
                                                    save_best_only=True,
                                                    save_freq='epoch',
                                                    )
    callbks.append(model_chpt1)

    # csv logger
    csv_save_path = os.path.join(output_folder, f'epoch_history_run{run_index}.csv')
    csv_logger = tf.keras.callbacks.CSVLogger(csv_save_path)
    callbks.append(csv_logger)

    start = time.time()

    history = model.fit(
        train_ds,
        epochs=300,
        validation_data=val_ds,
        callbacks=callbks
    )

    end = time.time()
    ellapsed_time = end - start

    encoder_save_path = os.path.join(output_folder, f'encoder_run{run_index}.h5')
    encoder.save(encoder_save_path)

    return model, history, ellapsed_time


def write_video_frames(avi_file, output_frames_path):
    cam = cv2.VideoCapture(avi_file)

    avi_name = os.path.basename(avi_file)
    avi_name = avi_name[:avi_name.rindex('.avi')]

    frameno = 0
    while (True):
        ret, frame = cam.read()
        if ret:
            frame_name = avi_name + f'_{str(frameno).zfill(4)}' + '.png'
            frame_path = os.path.join(output_frames_path, frame_name)

            if os.path.exists(frame_path):
                continue

            # if video is still left continue creating images

            cv2.imwrite(frame_path, frame)
            frameno += 1
        else:
            break

    cam.release()
    cv2.destroyAllWindows()


def prepare_video_data(data_folder):
    output_images_folder = r'Data/Unity-Classification-A-Frames'
    #    return

    if not os.path.exists(output_images_folder):
        os.makedirs(output_images_folder)

    dirs_dict = {
        'PLAX_full': r'Data/UNITY-Classification-A/1 PLAX/1 PLAX_full',
        'PLAX_valves': r'Data/UNITY-Classification-A/1 PLAX/2 PLAX_valves',
        'PLAX_PV': r'Data/UNITY-Classification-A/1 PLAX/3 PLAX_PV',
        'PLAX_TV': r'Data/UNITY-Classification-A/1 PLAX/4 PLAX_TV',

        'PSAX_AV': r'Data/UNITY-Classification-A/2 PSAX/1 PSAX_AV',
        'PSAX_LV': r'Data/UNITY-Classification-A/2 PSAX/2 PSAX_LV',
        'PSAX_PV': r'Data/UNITY-Classification-A/2 PSAX/3 PSAX_PV',

        'MV_LA_IAS': r'Data/UNITY-Classification-A/3 Apical/1 MV_LA_IAS',
        'A2CH': r'Data/UNITY-Classification-A/3 Apical/2 A2CH',
        'A3CH': r'Data/UNITY-Classification-A/3 Apical/3 A3CH',
        'A4CH_LV': r'Data/UNITY-Classification-A/3 Apical/4 A4CH/1 A4CH_LV',
        'A4CH_RV': r'Data/UNITY-Classification-A/3 Apical/4 A4CH/2 A4CH_RV',
        'A5CH': r'Data/UNITY-Classification-A/3 Apical/5 A5CH',
        'Apex': r'Data/UNITY-Classification-A/3 Apical/6 Apex',

        'Subcostal_heart': r'Data/UNITY-Classification-A/4 Subcostal/1 Subcostal_heart',
        'Subcostal_IVC': r'Data/UNITY-Classification-A/4 Subcostal/2 Subcostal_IVC',

        'Suprasternal': r'Data/UNITY-Classification-A/5 Suprasternal',
    }

    counts_dict = {}
    for class_name, value in dirs_dict.items():
        files = os.listdir(value)

        counts_dict[class_name] = len(files)

        for file in files:
            avi_file = os.path.join(value, file)
            output_path = os.path.join(output_images_folder, class_name)
            if not os.path.exists(output_path):
                os.mkdir(output_path)

            write_video_frames(avi_file, output_path)

    print(counts_dict)
    return

def get_second_labels(second_labels_file, test_files, test_labels, class_lookup):
    rev_lookup = dict((v, k) for k, v in class_lookup.items())
    test_labels2 = []
    file_class_dict = {}
    with open(second_labels_file) as file:
        for line in file:
            vid = line.rstrip()
            video_name = vid[vid.rindex('/') + 1:vid.rindex('.')]
            class_name = vid[:vid.rindex('.')]
            class_name = class_name[:class_name.rindex('/')]

            if '/' in class_name:
                class_name = class_name[class_name.rindex('/') + 1:]

            if ' ' in class_name:
                class_name = class_name[class_name.rindex(' ') + 1:]

            class_index = -1
            if class_name in rev_lookup:
                class_index = rev_lookup[class_name]
            elif class_name.lower() in rev_lookup:
                class_index = rev_lookup[class_name.lower()]
            elif class_name.upper() in rev_lookup:
                class_index = rev_lookup[class_name.upper()]
            elif class_name == 'PSAX_AV_PV':
                class_index = rev_lookup['parastnl_av_pv']
            else:
                print(f'Error: Cannot find matching class name: {class_name}')

            file_class_dict[video_name] = class_index

    #The test set containes abit more files than the files for which we have two sets of annotations.
    #In this case, if we do not have alternate label for a file, we just use the same label.
    second_labels_count = 0
    second_label_videos = []
    for i in range(len(test_files)):
        vid = test_files[i]
        video_name = vid[vid.rindex('/') + 1:]

        #set to default class index
        second_class_index = test_labels[i]
        #Use second label if there is
        if video_name in file_class_dict:
            second_class_index = file_class_dict[video_name]
            second_labels_count+=1
            second_label_videos.append(vid)

        test_labels2.append(second_class_index)


    return np.asarray(test_labels2), second_label_videos, second_labels_count


def draw_plots(save_folder, epochs_to_test):

    dirs = os.listdir(save_folder)

    linear_eval_dict = {}
    fine_tuning_dict = {}

    for dir in dirs:
        exp_dir = os.path.join(save_folder, dir)

        if 'plot' in dir:
            continue

        # if not 'supcon' in dir:
        #     if not 'imagenetTrue' in dir:
        #         continue

        results_dir = os.listdir(exp_dir)

        linear_epochs = []
        linear_scores = []
        ft_epochs = []
        ft_scores = []
        for res_folder in results_dir:
            path = os.path.join(exp_dir, res_folder)
            if 'linear' in res_folder:
                epoch = int(res_folder[:res_folder.index('_')])
                if epoch not in epochs_to_test:
                    continue
                results_json = utils.load_json(os.path.join(path, 'results.json'))
                acc = float(results_json["acc"][0])
                linear_epochs.append(epoch)
                linear_scores.append(acc)
            elif 'fine_tuning' in res_folder:
                epoch = int(res_folder[:res_folder.index('_')])
                if epoch not in epochs_to_test:
                    continue
                results_json = utils.load_json(os.path.join(path, 'results.json'))
                acc = float(results_json["acc"][0])
                ft_epochs.append(epoch)
                ft_scores.append(acc)
                pass

        linear_epochs, linear_scores = zip(*sorted(zip(linear_epochs, linear_scores)))
        ft_epochs, ft_scores = zip(*sorted(zip(ft_epochs, ft_scores)))

        exp_name = dir[45:]
        linear_eval_dict[exp_name] = [linear_epochs, linear_scores]
        fine_tuning_dict[exp_name] = [ft_epochs, ft_scores]

    colour_list = ['cornflowerblue', 'salmon',  'orchid', 'mediumseagreen', 'orange', 'yellowgreen']
    label_size = 40  # 36 #30
    tick_size = 16  # 32 #26
    inset_tick_size = 34  # 28 #22
    legend_font_size = 20  # 26

    plt.clf()
    fig, ax = plt.subplots(figsize=(18, 14), dpi=300)
    cnt=0
    for key, value in linear_eval_dict.items():
        epochs = value[0]
        scores = value[1]
        colour = colour_list[cnt]
        linestyle = 'dashed'
        plt.plot([str(ep) for ep in epochs], scores, marker='o', color=colour, label=f'{key}', linestyle=linestyle,
                 mew=2,
                 linewidth=4, markersize=12)
        cnt+=1

    plt.xlabel('Pre-training Epochs', fontsize=label_size, labelpad=20)
    plt.ylabel('Accuracy (Downstream)', fontsize=label_size)
    plt.xticks(fontsize=tick_size, ticks=[str(ep) for ep in epochs], labels=[str(ep) for ep in epochs])
    plt.yticks(fontsize=tick_size)

    # Set current axis
    # Because plt.axes adds an Axes to the current figure and makes it the current Axes.
    # To set the current axes, where ax is the Axes object you'd like to become active:

    # PLOT LEGEND
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.90])
    legend = plt.legend(fontsize=legend_font_size, loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=3)

    plt.savefig(os.path.join(save_folder, 'linear_eval_plot.png'), dpi=300, bbox_inches='tight')

    plt.clf()
    fig, ax = plt.subplots(figsize=(18, 14), dpi=300)
    cnt = 0
    for key, value in fine_tuning_dict.items():
        epochs = value[0]
        scores = value[1]
        colour = colour_list[cnt]
        linestyle = 'dashed'
        plt.plot([str(ep) for ep in epochs], scores, marker='o', color=colour, label=f'{key}', linestyle=linestyle,
                 mew=2,
                 linewidth=4, markersize=12)
        cnt += 1

    plt.xlabel('Pre-training Epochs', fontsize=label_size, labelpad=20)
    plt.ylabel('Accuracy (Downstream)', fontsize=label_size)
    plt.xticks(fontsize=tick_size, ticks=[str(ep) for ep in epochs], labels=[str(ep) for ep in epochs])
    plt.yticks(fontsize=tick_size)

    # Set current axis
    # Because plt.axes adds an Axes to the current figure and makes it the current Axes.
    # To set the current axes, where ax is the Axes object you'd like to become active:
    plt.sca(ax)

    # PLOT LEGEND
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.90])
    legend = plt.legend(fontsize=legend_font_size, loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=3)

    plt.savefig(os.path.join(save_folder, 'fine_tuning_plot.png'), dpi=300, bbox_inches='tight')


def remove_layers(model, remove_indices):
    """
    Rebuilds the given Keras model after removing specified layers.

    Parameters:
    - model: tf.keras.Model, the original model
    - remove_indices: list of layer indices to remove (0-based indexing)

    Returns:
    - new_model: tf.keras.Model, the modified model without the specified layers
    """
    # Get the original model's input
    inputs = model.input
    x = inputs

    # Iterate through the layers while skipping the specified ones
    for i, layer in enumerate(model.layers):
        if i not in remove_indices:
            x = layer(x)

    # Create a new model with the updated architecture
    new_model = tf.keras.Model(inputs=inputs, outputs=x)

    return new_model


def evaluate(save_folder, class_lookup, model_file, test_files, test_labels, run):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    test_images_ds = tf.data.Dataset.from_tensor_slices((test_files))
    test_images_ds = test_images_ds.map(process_video_only, num_parallel_calls=tf.data.AUTOTUNE)
    test_images_ds = test_images_ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    model = tf.keras.models.load_model(model_file, custom_objects={'VideoAug': VideoAug, 'RollingLayer': RollingLayer,
                                                                   'RandomDropLayer': RandomDropLayer})

    #remove the custom augmentation layer and rolling layer when testing
    # remove rolling layer only:

    y_pred = model.predict(test_images_ds)
    y_true = test_labels
    y_pred = y_pred.argmax(axis=-1)

    # Compute accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f" Accuracy run{run}: {acc}")
    # This gives metrics per class
    precision, recall, f1score, support = score(y_true, y_pred)
    # Average all
    precision_avg = np.mean(precision)
    recall_avg = np.mean(recall)
    f1_avg = np.mean(f1score)

    test_ds_1_sample = tf.data.Dataset.from_tensor_slices((test_files, test_labels))
    test_ds_1_sample = test_ds_1_sample.map(process_video_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds_1_sample = test_ds_1_sample.batch(1).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds_1_sample = test_ds_1_sample.take(1)
    start = time.time()
    test_1_sample = model.predict(test_ds_1_sample)
    end = time.time()

    results_dict = {}
    results_dict['num_classes'] = len(class_lookup)
    results_dict['acc'] = acc
    results_dict['precision'] = precision_avg
    results_dict['recall'] = recall_avg
    results_dict['f1'] = f1_avg
    results_dict['inference_time'] = end - start
    utils.write_dict_to_json(results_dict, save_folder, f'results_run{run}.json')

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix_2(cm, list(class_lookup.values()), normalize=False, title='Confusion Matrix')
    plt.savefig(os.path.join(save_folder, f'confusion_matrix_run{run}.png'))

    # Acc per class:
    acc_per_class = cm.diagonal() / cm.sum(axis=1)
    acc_per_class_dict = {}
    for idx, accy in enumerate(acc_per_class):
        acc_per_class_dict[class_lookup[idx]] = acc_per_class[idx]
    utils.write_dict_to_json(acc_per_class_dict, save_folder, f'classification_report_acc_run{run}.json')

    # Report
    class_report = classification_report(y_true, y_pred, target_names=list(class_lookup.values()), output_dict=True)
    utils.write_dict_to_json(class_report, save_folder, f'classification_report_run{run}.json')

def main():
    random.seed(444)

    global TOTAL_FRAMES
    if MIN_FRAMES % SKIP_FRAMES > 0:
        TOTAL_FRAMES += 1

    # CHANGE RUN SETTINGS HERE:
    train = True
    runs = 1
    merged = True
    linear_eval = False #Either Linear evaluation or full fine-tuning
    create_plots = False
    epochs_to_test = [25]#[25, 50, 100, 200, 400]  # [i for i in range(25, 525, 25)]

    second_labels_file = 'UNITY-Classification-A-DF-labels.txt'
    #The first set of labels are from the folder names and the folder names are the classes.
    #The second set of labels are saved in a text file with filepaths. The classes are extracted from the filepath.
    #The first set has 17 classes and the second set has 14 classes but the files are the same. The first set is just
    #more fine-grained than the second set. The applicable classes of the first set are merged so that it matches
    #the second set of labels.
    data_folder = r'Data/Unity-Classification-A-Frames'
    save_folder = f'video_class_evaluate_ssl_lstm_Final'
    evaluation_folder = 'video_class_diverse2_40min_Xception_epochs401_batch14_pervid5_imagenetTrue_noroll'

    if create_plots:
        draw_plots(save_folder, epochs_to_test)
        return

    if len(sys.argv) > 1:
        evaluation_folder = str(sys.argv[1])
        linear_eval = sys.argv[2]=='True'

    print(linear_eval)

    save_folder = os.path.join(save_folder, evaluation_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # temporary code below that overrides the above lists with specific train, val and test files.
    # This is so we can compare with the 2D_AVG results from video_classification_2D_average
    (train_files, train_labels,
     val_files, val_labels,
     test_files, test_labels,
     class_count, class_lookup,
     class_vid_count, excluded_cnt) = split_data_temp(data_folder=data_folder,
                                                      train_txt='video_classification_2D_40_frames_Xception_True_classes_14_8bx4g/train_files.txt',
                                                      val_txt='video_classification_2D_40_frames_Xception_True_classes_14_8bx4g/val_files.txt',
                                                      test_txt='video_classification_2D_40_frames_Xception_True_classes_14_8bx4g/test_files.txt',
                                                      merge=merged)

    test_labels_2nd, second_label_videos, second_labels_count = get_second_labels(second_labels_file,
                                                                                  test_files, test_labels,
                                                                                  class_lookup)

    y_pred = test_labels_2nd.tolist()
    y_true = test_labels
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1score, support = score(y_true, y_pred)
    print(f"Expert1 ref Accuracy: {acc}, f1score: {np.mean(f1score)}, precision: {np.mean(precision)}, recall: {np.mean(recall)}")

    acc = accuracy_score(y_pred, y_true)
    precision, recall, f1score, support = score(y_pred, y_true)
    print(f"Expert2 ref Accuracy: {acc}, f1score: {np.mean(f1score)}, precision: {np.mean(precision)}, recall: {np.mean(recall)}")

    print(f'num_train: {len(train_files)}')
    print(f'num_val: {len(val_files)}')
    print(f'num_test: {len(test_files)}')

    with open(os.path.join(save_folder, f'train_files.txt'), 'w') as f:
        for line in train_files:
            f.write(f"{line}\n")
    with open(os.path.join(save_folder, f'val_files.txt'), 'w') as f:
        for line in val_files:
            f.write(f"{line}\n")
    with open(os.path.join(save_folder, f'test_files.txt'), 'w') as f:
        for line in test_files:
            f.write(f"{line}\n")

    print(f'class_count: {class_count}')

    (train_ds, val_ds,
     test_ds, test_images_ds) = get_tf_datasets(train_files, train_labels,
                                                val_files, val_labels,
                                                test_files, test_labels, True)

    #     break

    unique, counts = np.unique(train_labels, return_counts=True)
    unique1, counts1 = np.unique(val_labels, return_counts=True)
    unique2, counts2 = np.unique(test_labels, return_counts=True)
    # Should all be equal:
    print(f'Number of classes in training set: {len(unique)}')
    print(f'Number of classes in val set: {len(unique1)}')
    print(f'Number of classes in test set: {len(unique2)}')

    train_set_counts_dict = {}
    val_set_counts_dict = {}
    test_set_counts_dict = {}
    for i in range(len(unique2)):
        train_set_counts_dict[class_lookup[i]] = int(counts[i])
        val_set_counts_dict[class_lookup[i]] = int(counts1[i])
        test_set_counts_dict[class_lookup[i]] = int(counts2[i])
    utils.write_dict_to_json(train_set_counts_dict, save_folder, f'per_class_count_train_set.json')
    utils.write_dict_to_json(val_set_counts_dict, save_folder, f'per_class_count_val_set.json')
    utils.write_dict_to_json(test_set_counts_dict, save_folder, f'per_class_count_test_set.json')

    global NUM_CLASSES
    NUM_CLASSES = len(unique)


    for k in range(len(epochs_to_test)):
        ssl_model_file = os.path.join(evaluation_folder, f'model_ep_{epochs_to_test[k]}.h5')
        output_folder = os.path.join(save_folder, f'{epochs_to_test[k]}_fine_tuning_results')
        if linear_eval:
            output_folder = os.path.join(save_folder, f'{epochs_to_test[k]}_linear_eval_results')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        acc_avg_runs, acc_avg_runs2 = [], []
        precision_avg_runs, precision_avg_runs2 = [], []
        recall_avg_runs, recall_avg_runs2 = [], []
        f1_avg_runs, f1_avg_runs2 = [], []
        run_losses, run_epochs = [], []
        run_val_losses, run_val_epochs = [], []
        training_times = []
        inference_times = []

        acc_avg_runs_2nd_labels, acc_avg_runs_2nd_labels2 = [], []
        precision_avg_runs_2nd_labels, precision_avg_runs_2nd_labels2 = [], []
        recall_avg_runs_2nd_labels, recall_avg_runs_2nd_labels2 = [], []
        f1_avg_runs_2nd_labels, f1_avg_runs_2nd_labels2 = [], []

        for i in range(runs):

            model = None
            training_info={}
            if train:
                # train from scratch or fine-tune happens here
                model, history, ellapsed_time = train_model(train_ds=train_ds, val_ds=val_ds,
                                                             output_folder=output_folder,
                                                             run_index=i,
                                                             num_train=len(train_files),
                                                             model_file = ssl_model_file,
                                                             linear_eval = linear_eval)

                loss = np.array(history.history['loss']).astype(float)
                val_loss = np.array(history.history['val_loss']).astype(float)
                run_losses.append(np.min(loss))
                run_epochs.append(int(np.argmin(loss)) + 1)  # 0-based index but epoch is 1-based
                run_val_losses.append(np.min(val_loss))
                run_val_epochs.append(int(np.argmin(val_loss)) + 1)
                training_times.append(ellapsed_time)

                training_info['run_losses'] = run_losses
                training_info['run_epochs'] = run_epochs
                training_info['run_val_losses'] = run_losses
                training_info['run_val_epochs'] = run_epochs
                training_info['training_times'] = training_times
                utils.write_dict_to_json(training_info, output_folder, f'training_info.json')

            tf.keras.backend.clear_session()

            # Get existing weights from previous training and make predictions on test dataset
            model_file = os.path.join(output_folder, f'model_run{i}.h5')
            evaluate(os.path.join(output_folder, 'results'), class_lookup, model_file, test_files, test_labels, i)

            model_file2 = os.path.join(output_folder, f'model_run{i}_val_acc.h5')
            evaluate(os.path.join(output_folder, 'results_valaccstop'), class_lookup, model_file2, test_files,
                     test_labels, i)

            # measure accuracy using another annotators labels (D-labels):
            evaluate(os.path.join(output_folder, 'results_2ndLabels'), class_lookup, model_file, test_files,
                     test_labels_2nd, i)
            evaluate(os.path.join(output_folder, 'results_2ndLabels_valaccstop'), class_lookup, model_file2, test_files,
                     test_labels_2nd, i)

            #plot learning curve

    print('Completed.')


if __name__ == "__main__":
    main()
