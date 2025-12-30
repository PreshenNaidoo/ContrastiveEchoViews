"""Video classification (supervised contrastive / SupCon + LSTM downstream)."""

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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

import architecture_ResNet
import utils

IMAGE_SIZE = (224, 224, 3)
BATCH_SIZE = 14#24#14#14 # 1 for a single video at a time
BACKBONES = ['Resnet50', 'Xception', 'DenseNet121']
NUM_CLASSES = 17  # 53 #This will be merged(in code) later on to 14 classes
MIN_FRAMES = 40 #Select all videos with minimum 40 frames
SKIP_FRAMES = 8#4#8 # =4 means select every 4th frame in a video
TOTAL_FRAMES = MIN_FRAMES//SKIP_FRAMES
GRAYSCALE = False
EPOCHS = 401

NETWORK_WIDTH = 128
TEMPERATURE = 0.5

AUG_MIN_AREA = 0.8
AUG_ROTATION = 0.10

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
        """Init.
        
        Args:
            learning_rate_base: Parameter.
            total_steps: Parameter.
            warmup_learning_rate: Parameter.
            warmup_steps: Parameter.
        
        Returns:
            None: Return value.
        """
        """Init.
        
        Args:
            learning_rate_base: Parameter.
            total_steps: Parameter.
            warmup_learning_rate: Parameter.
            warmup_steps: Parameter.
        
        Returns:
            None: Return value.
        """
        super(WarmUpCosine, self).__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        """Call.
        
        Args:
            step: Parameter.
        
        Returns:
            object: Return value.
        """
        """Call.
        
        Args:
            step: Parameter.
        
        Returns:
            object: Return value.
        """
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
    """Split the dataset into train/validation/test video lists and labels.
    
    Args:
        data_folder: Parameter.
        perc_train: Parameter.
        perc_val_of_train: Parameter.
        perc_test: Parameter.
        merge: Parameter.
    
    Returns:
        tuple | list: Return value.
    """
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
    """Split the dataset into train/validation/test video lists and labels.
    
    Args:
        data_folder: Parameter.
        train_txt: Parameter.
        val_txt: Parameter.
        test_txt: Parameter.
        merge: Parameter.
    
    Returns:
        tuple | list: Return value.
    """
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
    """Process video only.
    
    Args:
        video_file: Parameter.
    
    Returns:
        object: Return value.
    """
    frames = []
    for i in range(0, MIN_FRAMES, SKIP_FRAMES):
        frame = tf.strings.as_string(f'_{str(i).zfill(4)}.png')
        image_file = tf.strings.join([video_file, frame])
        img = tf.io.read_file(image_file)
        image = tf.image.decode_png(img, channels=3)

        if GRAYSCALE:
            image = tf.image.rgb_to_grayscale(image)

        h, w = image.shape[:2]

        if h != IMAGE_SIZE[0] and w != IMAGE_SIZE[1]:
            image = tf.image.resize(image, (IMAGE_SIZE[0], IMAGE_SIZE[1]))

        frames.append(image)

    frames_tensor = tf.stack(frames, 0)

    return frames_tensor


def process_video_and_label(video_file, label):
    """Process video and label.
    
    Args:
        video_file: Parameter.
        label: Parameter.
    
    Returns:
        tuple | list: Return value.
    """
    return process_video_only(video_file), label


def get_augmenter_echo(image_size, min_area, rotation):
    """Get augmenter echo.
    
    Args:
        image_size: Target size.
        min_area: Parameter.
        rotation: Parameter.
    
    Returns:
        object: Return value.
    """
    zoom_factor = 1.0 - math.sqrt(min_area)
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=image_size),
            tf.keras.layers.RandomTranslation(zoom_factor / 2, zoom_factor / 2),
            tf.keras.layers.RandomZoom((-zoom_factor, 0.0), (-zoom_factor, 0.0)),
            tf.keras.layers.RandomRotation(rotation)
        ]
    )


def get_tf_datasets(train_files, train_labels,
                    val_files, val_labels,
                    test_files, test_labels, augmentation=False):
    """Get tf datasets.
    
    Args:
        train_files: Parameter.
        train_labels: Parameter.
        val_files: Parameter.
        val_labels: Parameter.
        test_files: Parameter.
        test_labels: Parameter.
        augmentation: Parameter.
    
    Returns:
        tuple | list: Return value.
    """
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
        """Init.
        
        Args:
            numframes: Parameter.
            height: Parameter.
            width: Parameter.
            num_channels: Parameter.
            min_area: Parameter.
            rotation: Parameter.
            name: Parameter.
            kwargs: Additional keyword arguments.
        
        Returns:
            None: Return value.
        """
        """Init.
        
        Args:
            numframes: Parameter.
            height: Parameter.
            width: Parameter.
            num_channels: Parameter.
            min_area: Parameter.
            rotation: Parameter.
            name: Parameter.
            kwargs: Additional keyword arguments.
        
        Returns:
            None: Return value.
        """
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
        """Get config.
        
        Returns:
            object: Return value.
        """
        """Get config.
        
        Returns:
            object: Return value.
        """
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
    """Tf shuffle axis.
    
    Args:
        value: Parameter.
        axis: Parameter.
        seed: Parameter.
        name: Parameter.
    
    Returns:
        object: Return value.
    """
    perm = list(tf.range(tf.rank(value)))
    perm[axis], perm[0] = perm[0], perm[axis]
    value = tf.random.shuffle(tf.transpose(value, perm=perm))
    value = tf.transpose(value, perm=perm)
    return value

class RollingLayer(tf.keras.layers.Layer):
    def __init__(self):
        """Init.
        
        Returns:
            None: Return value.
        """
        """Init.
        
        Returns:
            None: Return value.
        """
        super(RollingLayer, self).__init__()

    def build(self, input_shape):
        """Build.
        
        Args:
            input_shape: Parameter.
        
        Returns:
            None: Return value.
        """
        """Build.
        
        Args:
            input_shape: Parameter.
        
        Returns:
            None: Return value.
        """
        super(RollingLayer, self).build(input_shape)

    def call(self, inputs):
        """Call.
        
        Args:
            inputs: Parameter.
        
        Returns:
            object: Return value.
        """
        """Call.
        
        Args:
            inputs: Parameter.
        
        Returns:
            object: Return value.
        """
        rand = tf.random.uniform(shape=(), minval=1, maxval=(TOTAL_FRAMES)-1, dtype=tf.int32)
        rolled = tf.roll(inputs, rand, axis=1) #skip batch dim
        return rolled

class RandomDropLayer(tf.keras.layers.Layer):
    def __init__(self, rows):
        """Init.
        
        Args:
            rows: Parameter.
        
        Returns:
            None: Return value.
        """
        """Init.
        
        Args:
            rows: Parameter.
        
        Returns:
            None: Return value.
        """
        super(RandomDropLayer, self).__init__()
        self.rows = rows

    def build(self, input_shape):
        """Build.
        
        Args:
            input_shape: Parameter.
        
        Returns:
            None: Return value.
        """
        """Build.
        
        Args:
            input_shape: Parameter.
        
        Returns:
            None: Return value.
        """
        super(RandomDropLayer, self).build(input_shape)


    def call(self, inputs):
        """Call.
        
        Args:
            inputs: Parameter.
        
        Returns:
            object: Return value.
        """
        """Call.
        
        Args:
            inputs: Parameter.
        
        Returns:
            object: Return value.
        """
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


import enum


@enum.unique
class ModelMode(enum.Enum):
    TRAIN = 1
    EVAL = 2
    INFERENCE = 3


@enum.unique
class AugmentationType(enum.Enum):
    """Valid augmentation types."""
    SIMCLR = 's'
    AUTOAUGMENT = 'a'
    RANDAUGMENT = 'r'
    # SimCLR combined with RandAugment.
    STACKED_RANDAUGMENT = 'sr'
    # No augmentation.
    IDENTITY = 'i'


@enum.unique
class LossContrastMode(enum.Enum):
    ALL_VIEWS = 'a'  # All views are contrasted against all other views.
    ONE_VIEW = 'o'  # Only one view is contrasted against all other views.


@enum.unique
class LossSummationLocation(enum.Enum):
    OUTSIDE = 'o'  # Summation location is outside of logarithm
    INSIDE = 'i'  # Summation location is inside of logarithm


@enum.unique
class LossDenominatorMode(enum.Enum):
    ALL = 'a'  # All negatives and all positives
    ONE_POSITIVE = 'o'  # All negatives and one positive
    ONLY_NEGATIVES = 'n'  # Only negatives


@enum.unique
class Optimizer(enum.Enum):
    RMSPROP = 'r'
    MOMENTUM = 'm'
    LARS = 'l'
    ADAM = 'a'
    NESTEROV = 'n'


@enum.unique
class EncoderArchitecture(enum.Enum):
    RESNET_V1 = 'r1'
    RESNEXT = 'rx'


@enum.unique
class DecayType(enum.Enum):
    COSINE = 'c'
    EXPONENTIAL = 'e'
    PIECEWISE_LINEAR = 'p'
    NO_DECAY = 'n'


@enum.unique
class EvalCropMethod(enum.Enum):
    """Methods of cropping eval images to the target dimensions."""
    # Resize so that min image dimension is IMAGE_SIZE + CROP_PADDING, then crop
    # the central IMAGE_SIZExIMAGE_SIZE square.
    RESIZE_THEN_CROP = 'rc'
    # Crop a central square of side length
    # IMAGE_SIZExIMAGE_SIZE.
    CROP_THEN_RESIZE = 'cr'
    # Crop the central IMAGE_SIZE/(IMAGE_SIZE+CROP_PADDING) pixels along each
    # dimension, preserving the natural image aspect ratio, then resize to
    # IMAGE_SIZExIMAGE_SIZE, which distorts the image.
    CROP_THEN_DISTORT = 'cd'
    # Do nothing. Requires that the input image is already the desired size.
    IDENTITY = 'i'

# Define the contrastive model with model-subclassing
class Contrastive_Model(tf.keras.Model):
    def __init__(self, encoder):
        """Init.
        
        Args:
            encoder: Backbone encoder network.
        
        Returns:
            None: Return value.
        """
        """Init.
        
        Args:
            encoder: Backbone encoder network.
        
        Returns:
            None: Return value.
        """
        super().__init__()

        self.temperature = TEMPERATURE
        self.encoder = encoder
        # Non-linear MLP as projection head
        #Get output shape of encoder

        self.roller = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(TOTAL_FRAMES, IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2])),
            RollingLayer()
        ])

        #20 epochs

        # 30 epochs

        self.vid_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(TOTAL_FRAMES, IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2])),
            tf.keras.layers.Rescaling(1. / 255.0),
            tf.keras.layers.TimeDistributed(encoder),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(128, return_sequences=False, recurrent_dropout=0.5, dropout=0.1)),
            tf.keras.layers.Flatten(),
        ])

        self.temp = self.vid_model.output_shape
        self.sub_model_output_shape = (self.vid_model.output_shape[1])

        self.projection_head = tf.keras.Sequential(
        [
                tf.keras.Input(shape=self.sub_model_output_shape),  # output shape of the encoder
                tf.keras.layers.Dense(NETWORK_WIDTH, activation="relu"),
                tf.keras.layers.Dense(NETWORK_WIDTH),
            ],
            name="Projection_Head",
        )

        self.vid_model.summary()
        self.projection_head.summary()

    def compile(self, contrastive_optimizer, **kwargs):
        """Configure the model for training.
        
        Args:
            contrastive_optimizer: Parameter.
            kwargs: Additional keyword arguments.
        
        Returns:
            None: Return value.
        """
        """Configure the model for training.
        
        Args:
            contrastive_optimizer: Parameter.
            kwargs: Additional keyword arguments.
        
        Returns:
            None: Return value.
        """
        super().compile(**kwargs)

        self.contrastive_optimizer = contrastive_optimizer


        self.contrastive_loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        """Return the list of tracked Keras metrics.
        
        Returns:
            tuple | list: Return value.
        """
        """Return the list of tracked Keras metrics.
        
        Returns:
            tuple | list: Return value.
        """
        return [
            self.contrastive_loss_tracker,
        ]

    def _cap_positives_mask(self, untiled_mask, diagonal_mask, num_views, positives_cap):
        r"""Cap positives in the provided untiled_mask.

            'positives_cap' specifies the maximum number of positives *other* than
            augmentations of the anchor. Positives will be evenly sampled from all
            views.

        Args:
          untiled_mask: Tensor of shape [local_batch_size, global_batch_size] that has
            entry (r, c) == 1 if feature entries in rows r and c are from the same
            class. Else (r, c) == 0.
          diagonal_mask: Tensor with the same shape as `untiled_mask`. When
            local_batch_size == global_batch_size this is just an identity matrix.
            Otherwise, it is an identity matrix of size `local_batch_size` that is
            padded with 0's in the 2nd dimension to match the target shape. This is
            used to indicate where the anchor views exist in the global batch of
            views.
          num_views: Integer number of total views.
          positives_cap: Integer maximum number of positives *other* than
            augmentations of anchor. Infinite if < 0. Must be multiple of num_views.
            Including augmentations, a maximum of (positives_cap + num_views - 1)
            positives is possible. This parameter modifies the contrastive numerator
            by selecting which positives are present in the summation, and which
            positives contribure to the denominator if denominator_mode ==
            enums.LossDenominatorMode.ALL.

        Returns:
          A tf.Tensor with the modified `untiled_mask`.
        """
        untiled_mask_no_diagonal = tf.math.minimum(untiled_mask, 1. - diagonal_mask)
        untiled_positives_per_anchor = positives_cap // num_views

        # Grabs top-k positives from each row in the mask. Can end up with negatives
        # incorrectly marked as positives if fewer than `untiled_positives_per_anchor`
        # exist in any row of `untiled_mask_no_diagonal`. However, these false
        # positives wil be masked out before the function returns.
        _, top_k_col_idx = tf.math.top_k(untiled_mask_no_diagonal,
                                         untiled_positives_per_anchor)
        top_k_row_idx = tf.expand_dims(tf.range(tf.shape(untiled_mask)[0]), axis=1)

        # Construct |top_k_idx|, a tensor of shape
        # the 2D index in a
        # tensor which holds a positive; all others are negatives.
        top_k_idx = tf.reshape(
            tf.stack([
                tf.tile(top_k_row_idx,
                        (1, untiled_positives_per_anchor)), top_k_col_idx
            ],
                axis=-1), (-1, 2))

        # Construct |untiled_mask|. Sets positives to 1 according to top_k_idx
        # above.
        untiled_mask_capped = tf.scatter_nd(
            top_k_idx,
            tf.ones(
                shape=tf.shape(top_k_idx)[0], dtype=untiled_mask_no_diagonal.dtype),
            untiled_mask_no_diagonal.shape)
        untiled_mask_capped = tf.math.maximum(untiled_mask_capped, diagonal_mask)
        return untiled_mask * untiled_mask_capped

    def _create_tiled_masks(self, untiled_mask, diagonal_mask, num_views,
                            num_anchor_views, positives_cap):
        r"""Creates tiled versions of untiled mask.

        Tiles `untiled_mask`, which has shape [local_batch_size, global_batch_size]
        by factors of [num_anchor_views, num_views], and then generates two masks from
        it. In both cases, the mask dimensions are ordered by view and then by sample,
        so if there was a batch size of 3 with 2 views the order would be
        [b1v1, b2v1, b3v1, b1v2, b2v2, b3v2]:
          positives_mask: Entry (row = i, col = j) is 1 if
            untiled_mask[i % local_batch_size, j % global_batch_size] == 1 and
            i // local_batch_size != j // global_batch_size. This results in a mask
            that is 1 for all pairs that are the same class but are not the exact same
            view. An exception to this is if positives_cap > -1, in which case there
            is a maximum of (positives_cap) 1-values per row, not including the
            entries that correspond to other views of the anchor. That is,
            positives_cap does nothing if there is only a single 1-valued entry per
            row in `untiled_mask`.
          negatives_mask: Entry (row = i, col = j) is 1 if features i and j are
            different classes. Otherwise the entry is 0.

        Args:
          untiled_mask: Tensor of shape [local_batch_size, global_batch_size], where
            local_batch_size <= global_batch_size, that has entry (r, c) == 1 if
            feature entries in rows r and c are from the same class. Else (r, c) == 0.
            In the self-supervised case, where the only positives are other views of
            the same sample, `untiled_mask` and `diagonal_mask` should be the same.
          diagonal_mask: Tensor with the same shape as `untiled_mask`. When
            local_batch_size == global_batch_size this is just an identity matrix.
            Otherwise, it is a slice of a [global_batch_size, global_batch_size]
            identity matrix that indicates where in the global batch the local batch
            is located.
          num_views: Integer number of total views.
          num_anchor_views: Integer number of anchor views.
          positives_cap: Integer maximum number of positives *other* than
            augmentations of anchor. Infinite if < 0. Must be multiple of num_views.
            Including augmentations, a maximum of (positives_cap + num_views - 1)
            positives is possible. This parameter modifies the contrastive numerator
            by selecting which positives are present in the summation, and which
            positives contribure to the denominator if denominator_mode ==
            enums.LossDenominatorMode.ALL.

        Returns:
          Tuple containing positives_mask and negatives_mask tensors.
        """
        global_batch_size = tf.shape(untiled_mask)[1]
        # Generate |all_but_diagonal_mask|, a tensor of shape
        # corresponding to row i is the same as the view corresponding to column j. In
        labels = tf.argmax(diagonal_mask, axis=-1)
        tiled_labels = []
        for i in range(num_anchor_views):
            tiled_labels.append(labels + tf.cast(global_batch_size, labels.dtype) * i)
        tiled_labels = tf.concat(tiled_labels, axis=0)
        tiled_diagonal_mask = tf.one_hot(tiled_labels, global_batch_size * num_views)
        all_but_diagonal_mask = 1. - tiled_diagonal_mask

        # Construct |negatives_mask| and |uncapped_positives_mask|, both tensors of
        # |uncapped_positives_mask| are all positive candidates, including the
        # diagonal representing the anchor view itself, before the capping procedure.
        # Any element that is not an `uncapped` positive is a negative.
        uncapped_positives_mask = tf.tile(untiled_mask, [num_anchor_views, num_views])

        negatives_mask = 1. - uncapped_positives_mask

        # Select only 'positives_cap' positives by selecting top-k values of 0/1 mask
        # and scattering ones into those indices. This capping is done on only
        # non-diagonal positives.
        if positives_cap > -1:
            untiled_mask = self._cap_positives_mask(untiled_mask, diagonal_mask, num_views,
                                               positives_cap)
            # Construct |positives_mask|, a tensor of shape
            # r != c.
            # Else it is zero.
            positives_mask = tf.tile(untiled_mask, [num_anchor_views, num_views])
        else:
            positives_mask = uncapped_positives_mask

        positives_mask = positives_mask * all_but_diagonal_mask  # Zero the diagonal.

        return positives_mask, negatives_mask

    def _validate_contrastive_loss_inputs(self, features, labels, contrast_mode,
                                          summation_location, denominator_mode,
                                          positives_cap):
        r"""Validates inputs for contrastive_loss().

        Args:
          features: Tensor of rank at least 3, where the first 2 dimensions are
            batch_size and num_views, and the remaining dimensions are the feature
            shape.
          labels: One-hot labels tensor of shape [batch_size, num_labels] with numeric
            dtype.
          contrast_mode: LossContrastMode specifying which views get used as anchors.
          summation_location: LossSummationLocation specifying location of positives
            summation. See documentation above for more details.
          denominator_mode: LossDenominatorMode specifying which positives to include
            in contrastive denominator. See documentation above for more details.
          positives_cap: Integer maximum number of positives *other* than
            augmentations of anchor. Infinite if < 0. Must be multiple of num_views.
            Including augmentations, a maximum of (positives_cap + num_views - 1)
            positives is possible. This parameter modifies the contrastive numerator
            by selecting which positives are present in the summation, and which
            positives contribure to the denominator if denominator_mode ==
            enums.LossDenominatorMode.ALL.

        Returns:
          Tuple containing batch_size and num_views values.

        Raises:
          ValueError if any of the inputs are invalid.
        """
        if features.shape.rank < 3:
            raise ValueError(
                f'Invalid features rank ( = {features.shape.rank}). Should have rank '
                '>= 3 with shape [batch_size, num_views] + `feature shape.`')

        batch_size = tf.compat.dimension_at_index(features.shape, 0).value
        if batch_size is None:
            raise ValueError('features has unknown batch_size dimension.')
        num_views = tf.compat.dimension_at_index(features.shape, 1).value
        if num_views is None:
            raise ValueError('features has unknown num_views dimension.')

        #     # Check that |labels| are shaped like a one_hot vector.
        #         raise ValueError(

        if not isinstance(contrast_mode, LossContrastMode):
            raise ValueError(
                f'Invalid contrast_mode (= {contrast_mode}). Should be an instance of '
                'LossContrastMode.')
        if not isinstance(summation_location, LossSummationLocation):
            raise ValueError(
                f'Invalid summation_location (= {summation_location}). Should be an '
                'instance of LossSummationLocation.')
        if not isinstance(denominator_mode, LossDenominatorMode):
            raise ValueError(
                f'Invalid denominator_mode (= {denominator_mode}). Should be an '
                'instance of LossDenominatorMode.')
        if positives_cap > -1 and positives_cap % num_views != 0:
            raise ValueError(
                f'positives_cap (= {positives_cap}) must be a multiple of the '
                f'num_views (= {num_views}).')

        return batch_size, num_views

    def contrastive_loss(self, features,
                         labels=None,
                         temperature=1.0,
                         contrast_mode=LossContrastMode.ALL_VIEWS,
                         summation_location=LossSummationLocation.OUTSIDE,
                         denominator_mode=LossDenominatorMode.ALL,
                         positives_cap=-1,
                         scale_by_temperature=True):
        r"""Contrastive loss over features.

        Implemented as described in: https://arxiv.org/abs/2004.11362, Equation 2.

        Given `num_views` different views of each of `batch_size` samples, let `f_i`
        (i \in [1, 2 ... (num_views * batch_size)]) denote each respective feature
        vector. The contrastive loss then takes the following form:

          L = \sum_{i} L_i

        where each L_i is computed as:

          L_i = -\tau * \sum_{k \in P(i)} \log(p_{ik})    (1)

        where P(i) is the set of positives for entry i (distinct from i) and where:

                             \exp(f_i^T f_k / \tau)
          p_{ik} = ----------------------------------------                        (2)
                   \sum_{j \in A(i)} \exp(f_i^T f_j / \tau)

        where A(i) is the set of all positives or negatives (distinct from i). `i` is
        the anchor, and \tau is the temperature.

        This maximizes the likelihood of a given (anchor, positive) pair with
        respect to all possible pairs where the first member is the anchor and the
        second member is a positive or a negative.

        A typical way to define a positive is to define samples from the
        same class (but not the anchor itself) regardless of what view they are from.
        Similarly, a typical way to define a negative is for it to be any view of a
        sample from a different class.

        There are two ways to define which feature pairs should be treated as
        positives and negatives. All views of the same sample are always treated as
        positives. You can declare other samples to be positives by providing `labels`
        such that all samples with the same label will be positives for each other.

        If `labels` is not provided then we default to every sample belonging to its
        own unique class. Therefore, the only positive used is another view of the
        anchor itself. This implements the loss as described in:

          https://arxiv.org/pdf/2002.05709.pdf
          A Simple Framework for Contrastive Learning of Visual Representations
          Chen T., Kornblith S., Norouzi M., Hinton G.

        It is recommended to use features whose L_2 norm is 1. since that ensures
        that the loss does not return NaN values without changing the intended
        behaviour of the loss function.

        In (1) above, note that the summation over positives is located outside of the
        \log(). However, one can permute these two operations. The result is Eq. 3 in
        https://arxiv.org/abs/2004.11362. Users can specify the location of the
        summation relative to the \log() via the `summation_location' argmument:
         - 'out': Eq. 2 in https://arxiv.org/abs/2004.11362.
         - 'in' : Eq. 3 in https://arxiv.org/abs/2004.11362.

        Additionally, in (2) above, note that the denominator sums over *all* entries
        distinct from i. One can change which terms are included in the denominator
        via the `denominator_mode` argument:
         - LossDenominatorMode.ALL : All entries (i.e., all negatives and all
                   positives) distinct from i are included.
         - LossDenominatorMode.ONE_POSITIVE : All negatives are included but only the
                   single positive in the numerator of (2) is included. Any other
                   positives are excluded.
         - LossDenominatorMode.ONLY_NEGATIVES: All negatives are included but no
                   positives are, not even the single positive in the numerator of
                   (2).

        On TPUs, this method will internally perform the cross-replica operations that
        enable using the samples from all cores in computing the loss. The inputs to
        this function should be the features and labels from a single core and each
        core will compute the loss using just these features as anchors, but will use
        positives and negatives from the full global batch. Since the loss for each
        anchor is only computed on one TPU core, it's still necessary to have a
        cross-replica reduction in the final loss computation.

        Also, though it is not applicable to multiview contrastive learning, this
        function will work if |features| contains only 1 view. In the high batch size
        limit, the implemented contrastive loss with only 1 view, positives_cap = 1,
        and temperature = 1.0 is equivalent to the N-pairs loss
        (https://papers.nips.cc/paper/6200-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective.pdf)

        Args:
          features: A Tensor of rank at least 3, where the first 2 dimensions are
            batch_size and num_views, and the remaining dimensions are the feature
            shape. Note that when running on TPU, batch_size is the per-core batch
            size.
          labels: One-hot labels to be used to construct the supervised contrastive
            loss. Samples with the same labels are used as positives for each other.
            Labels must have shape [batch_size, num_labels] with numeric dtype and be
            0-1 valued. Note that when running on TPU, batch_size is the per-core
            batch size.
          temperature: Temperature at which softmax evaluation is done. Temperature
            must be a python scalar or scalar Tensor of numeric dtype.
          contrast_mode: LossContrastMode specifying which views get used as anchors
            (f_i in the expression above)
            'ALL_VIEWS': All the views of all samples are used as anchors (f_i in the
              expression above).
            'ONE_VIEW': Just the first view of each sample is used as an anchor (f_i
              in the expression above). This view is called the `core` view against
              which other views are contrasted.
          summation_location: LossSummationLocation specifying location of positives
            summation. See documentation above for more details.
          denominator_mode: LossDenominatorMode specifying which positives to include
            in contrastive denominator. See documentation above for more details.
          positives_cap: Integer maximum number of positives *other* than
            augmentations of anchor. Infinite if < 0. Must be multiple of num_views.
            Including augmentations, a maximum of (positives_cap + num_views - 1)
            positives is possible. This parameter modifies the contrastive numerator
            by selecting which positives are present in the summation, and which
            positives contribure to the denominator if denominator_mode ==
            enums.LossDenominatorMode.ALL.
          scale_by_temperature: Boolean. Whether to scale the loss by `temperature`.
            The loss gradient naturally has a 1/temperature scaling factor, so this
            counteracts it.

        Returns:
          Scalar tensor with contrastive loss value with shape [batch_size] and dtype
          tf.float32. The loss for each batch element is the mean over all views.

        Raises:
          ValueError if the shapes of any of the Tensors are unexpected, or if both
          `labels` and `mask` are not `None`.
        """
        features = tf.convert_to_tensor(features)
        labels = tf.convert_to_tensor(labels) if labels is not None else None

        local_batch_size, num_views = self._validate_contrastive_loss_inputs(
            features, labels, contrast_mode, summation_location, denominator_mode,
            positives_cap)

        # Flatten `features` to a single dimension per view per sample so it has shape
        if features.shape.rank > 3:
            features = tf.reshape(features,
                                  tf.concat([tf.shape(features)[:2], [-1]], axis=0),
                                  'flattened_features')
        if features.dtype != tf.float32:
            features = tf.cast(features, tf.float32)

        # Grab the features from all TPU cores. We use the local batch as anchors and
        # the full global batch as contrastives. If not on TPU, global_features is the
        # same as features.
        global_features = features#utils.cross_replica_concat(features)
        global_batch_size = tf.compat.dimension_at_index(global_features.shape,
                                                         0).value
        local_replica_id = 0#utils.local_tpu_replica_id()

        # the current replica.
        diagonal_mask = tf.one_hot(
            tf.range(local_batch_size) + (local_replica_id * local_batch_size),
            global_batch_size)

        # indicates which samples should be considered positives for each other.
        if labels is None:
            # Defaults to every sample belonging to its own unique class, containing
            # just that sample and other views of it.
            mask = diagonal_mask
        else:
            labels = tf.cast(labels, tf.float32)  # TPU matmul op unsupported for ints.
            global_labels = labels#utils.cross_replica_concat(labels)
            mask = tf.linalg.matmul(labels, global_labels, transpose_b=True)
        mask = tf.ensure_shape(mask, [local_batch_size, global_batch_size])

        # To streamline the subsequent TF, the first two dimensions of
        # transposed and then flattened. The result has shape
        # elements are grouped by view, not by sample.
        all_global_features = tf.reshape(
            tf.transpose(global_features, perm=[1, 0, 2]),
            [num_views * global_batch_size, -1])

        if contrast_mode == LossContrastMode.ONE_VIEW:
            anchor_features = features[:, 0]
            num_anchor_views = 1
        else:  # contrast_mode == enums.LossContrastMode.ALL_VIEWS
            # Reshape features to match how global_features is reshaped above.
            anchor_features = tf.reshape(
                tf.transpose(features, perm=[1, 0, 2]),
                [num_views * local_batch_size, -1])
            num_anchor_views = num_views

        # anchor features with all features. It has shape
        # improve numerical stability, subtract out the largest |logits| element in
        # each row from all elements in that row. Since |logits| is only ever used as
        # a ratio of exponentials of |logits| values, this subtraction does not change
        # the results correctness. A stop_gradient() is needed because this change is
        # just for numerical precision.
        logits = tf.linalg.matmul(
            anchor_features, all_global_features, transpose_b=True)
        temperature = tf.cast(temperature, tf.float32)
        logits = logits / temperature
        logits = (
                logits - tf.reduce_max(tf.stop_gradient(logits), axis=1, keepdims=True))
        exp_logits = tf.exp(logits)

        # The following masks are all tiled by the number of views, i.e., they have
        positives_mask, negatives_mask = (
            self._create_tiled_masks(mask, diagonal_mask, num_views, num_anchor_views,
                                positives_cap))
        num_positives_per_row = tf.reduce_sum(positives_mask, axis=1)

        if denominator_mode == LossDenominatorMode.ALL:
            denominator = tf.reduce_sum(
                exp_logits * negatives_mask, axis=1, keepdims=True) + tf.reduce_sum(
                exp_logits * positives_mask, axis=1, keepdims=True)
        elif denominator_mode == LossDenominatorMode.ONE_POSITIVE:
            denominator = exp_logits + tf.reduce_sum(
                exp_logits * negatives_mask, axis=1, keepdims=True)
        else:  # denominator_mode == enums.LossDenominatorMode.ONLY_NEGATIVES
            denominator = tf.reduce_sum(
                exp_logits * negatives_mask, axis=1, keepdims=True)

        # Note that num_positives_per_row can be zero only if 1 view is used. The
        if summation_location == LossSummationLocation.OUTSIDE:
            log_probs = (logits - tf.math.log(denominator)) * positives_mask
            log_probs = tf.reduce_sum(log_probs, axis=1)
            log_probs = tf.math.divide_no_nan(log_probs, num_positives_per_row)
        else:  # summation_location == enums.LossSummationLocation.INSIDE
            log_probs = exp_logits / denominator * positives_mask
            log_probs = tf.reduce_sum(log_probs, axis=1)
            log_probs = tf.math.divide_no_nan(log_probs, num_positives_per_row)
            log_probs = tf.math.log(log_probs)

        loss = -log_probs
        if scale_by_temperature:
            loss *= temperature
        loss = tf.reshape(loss, [num_anchor_views, local_batch_size])

        if num_views != 1:
            loss = tf.reduce_mean(loss, axis=0)
        else:
            # The 1 view case requires special handling bc, unlike in the > 1 view case,
            # not all samples are guaranteed to have a positive. Also, no reduction over
            # views is needed.
            num_valid_views_per_sample = (
                tf.reshape(num_positives_per_row, [1, local_batch_size]))
            loss = tf.squeeze(tf.math.divide_no_nan(loss, num_valid_views_per_sample))

        return loss

    def train_step(self, data):
        """Keras training/evaluation step implementation.
        
        Args:
            data: Input batch/dataset.
        
        Returns:
            object: Return value.
        """
        """Keras training/evaluation step implementation.
        
        Args:
            data: Input batch/dataset.
        
        Returns:
            object: Return value.
        """
        images = data #first axis is the batch

        # Each image is augmented twice, differently

        labels = data[1]

        aug_layer = VideoAug(TOTAL_FRAMES, IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2], AUG_MIN_AREA, AUG_ROTATION)
        aug_data = aug_layer(data[0])

        rolled_data = self.roller(aug_data) #augroll



        with tf.GradientTape() as tape:
            features_1 = self.vid_model(data[0], training=True)
            features_2 = self.vid_model(rolled_data, training=True)

            # The representations are passed through a projection mlp
            projections_1 = self.projection_head(features_1, training=True)
            projections_2 = self.projection_head(features_2, training=True)

            labels = tf.one_hot(labels, NUM_CLASSES) #tf.one_hot(self.labels, self.num_classes)

            contrastive_loss = self.contrastive_loss(
                tf.stack([projections_1, projections_2], axis=1),
                labels=labels,
                temperature=TEMPERATURE)

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

    #
    #     # Each image is augmented twice, differently
    #
    #
    #     # The representations are passed through a projection mlp
    #
    #

    def save_weights(self, filepath, overwrite=True):
        """Save weights.
        
        Args:
            filepath: File or directory path.
            overwrite: Parameter.
        
        Returns:
            None: Return value.
        """
        """Save weights.
        
        Args:
            filepath: File or directory path.
            overwrite: Parameter.
        
        Returns:
            None: Return value.
        """
        print(f'\n **SAVED WEIGHTS: f{filepath} \n')
        self.encoder.save_weights(filepath, overwrite)

    def save(self, filepath, overwrite=True, save_format=None, **kwargs):
        """Save the underlying encoder weights to disk.
        
        Args:
            filepath: File or directory path.
            overwrite: Parameter.
            save_format: Parameter.
            kwargs: Additional keyword arguments.
        
        Returns:
            None: Return value.
        """
        """Save the underlying encoder weights to disk.
        
        Args:
            filepath: File or directory path.
            overwrite: Parameter.
            save_format: Parameter.
            kwargs: Additional keyword arguments.
        
        Returns:
            None: Return value.
        """
        print(f'\n **SAVED MODEL: f{filepath} \n')
        self.vid_model.save(filepath, overwrite, save_format)

def train_model(train_ds, val_ds, output_folder, backbone, run_index, num_train, use_imagenet_weights):
    """Train model.
    
    Args:
        train_ds: Parameter.
        val_ds: Parameter.
        output_folder: Parameter.
        backbone: Parameter.
        run_index: Parameter.
        num_train: Parameter.
        use_imagenet_weights: Parameter.
    
    Returns:
        tuple | list: Return value.
    """
    model = None
    weight_file = os.path.join(os.path.join(output_folder, 'ssl'), 'encoder_model_weights.h5')
    model_file = os.path.join(os.path.join(output_folder, 'ssl'), 'encoder_model.h5')


    AUG_MIN_AREA = 0.8
    AUG_ROTATION = 0.05


    if not GRAYSCALE:
        if use_imagenet_weights:
            encoder = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=IMAGE_SIZE, pooling='avg')
        else:
            encoder = tf.keras.applications.Xception(weights=None, include_top=False, input_shape=IMAGE_SIZE, pooling='avg')
    else:
        encoder = tf.keras.applications.Xception(weights=None, include_top=False, input_shape=IMAGE_SIZE, pooling='avg')

    encoder.trainable = True
    encoder.summary()

    #


    # #COSINE DECAY
    #
    #
    # #EXPONENTIAL DECAY
    #
    #

    model = Contrastive_Model(encoder=encoder)
    model.compile(contrastive_optimizer=tf.keras.optimizers.Adam())


    callbks = []
    # early stopping

    #model checkpoint to save best weights
    model_save_path = os.path.join(output_folder, 'model_ep_{epoch}.h5')
    model_chpt = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,
                                                    monitor='loss',
                                                    verbose=1,
                                                    save_weights_only=False,
                                                    save_best_only=False,
                                                    save_freq='epoch',
                                                    period=25
                                                    )
    callbks.append(model_chpt)

    # csv logger
    csv_save_path = os.path.join(output_folder, f'epoch_history.csv')
    csv_logger = tf.keras.callbacks.CSVLogger(csv_save_path)
    callbks.append(csv_logger)

    start = time.time()

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        callbacks=callbks
    )

    end = time.time()
    ellapsed_time = end - start

    model_weights_save_path_last_epoch = os.path.join(output_folder, 'ssl_encoder_model_weights.h5')
    model.encoder.save_weights(model_weights_save_path_last_epoch)
    model_save_path_last_epoch = os.path.join(output_folder, 'ssl_encoder_model.h5')
    model.encoder.save(model_save_path_last_epoch)

    model_weights_save_path_last_epoch = os.path.join(output_folder, 'ssl_vid_model_weights.h5')
    model.vid_model.save_weights(model_weights_save_path_last_epoch)
    model_save_path_last_epoch = os.path.join(output_folder, 'ssl_vid_model.h5')
    model.vid_model.save(model_save_path_last_epoch)

    return model, history, ellapsed_time


def write_video_frames(avi_file, output_frames_path):
    """Run a predefined set of comparisons and write plots to disk.
    
    Args:
        avi_file: Parameter.
        output_frames_path: File or directory path.
    
    Returns:
        None: Return value.
    """
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


            cv2.imwrite(frame_path, frame)
            frameno += 1
        else:
            break

    cam.release()
    cv2.destroyAllWindows()


def prepare_video_data(data_folder):
    """Prepare video data.
    
    Args:
        data_folder: Parameter.
    
    Returns:
        None: Return value.
    """
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

def plot_tsne(predictions, color_list, labels_list, save_file):
    """Create and display a plot for the requested metric(s).
    
    Args:
        predictions: Parameter.
        color_list: Parameter.
        labels_list: Parameter.
        save_file: Parameter.
    
    Returns:
        tuple | list: Return value.
    """
    tsne = TSNE(n_components=2)
    trans_dim = tsne.fit_transform(predictions)
    x = trans_dim[0:, 0]
    y = trans_dim[0:, 1]

    plt.clf()
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot()
    scatter = ax.scatter(x, y, c=color_list, s=1.5)
    plt.legend(handles=scatter.legend_elements()[0], labels=labels_list)
    plt.tight_layout()
    plt.savefig(save_file, dpi=300, bbox_inches='tight')

    return x,y


def plot_pca(predictions, color_list, labels_list, save_file):
    """Create and display a plot for the requested metric(s).
    
    Args:
        predictions: Parameter.
        color_list: Parameter.
        labels_list: Parameter.
        save_file: Parameter.
    
    Returns:
        None: Return value.
    """
    pca = PCA(n_components=2)
    trans_dim = pca.fit_transform(predictions)
    x = trans_dim[0:, 0]
    y = trans_dim[0:, 1]

    plt.clf()
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot()
    scatter = ax.scatter(x, y, label=labels_list, c=color_list, s=1.5)
    plt.legend(handles=scatter.legend_elements()[0], labels=labels_list)
    plt.tight_layout()
    plt.savefig(save_file, dpi=300, bbox_inches='tight')

def select_random_files_from_list(files_list, num_to_select):
    """Select random files from list.
    
    Args:
        files_list: Parameter.
        num_to_select: Parameter.
    
    Returns:
        object: Return value.
    """
    select_files = []
    used_index = {}
    for i in range(num_to_select*100):
        index = random.randint(0, len(files_list)-1)
        if index not in used_index:
            select_files.append(files_list[index])
        used_index[index] = True
        if len(select_files) == num_to_select:
            break
    return select_files

def save_cosine_sim_heatmap(model, class_indices, lookup_label_dict, select_count, labels, save_name):
    """Save cosine sim heatmap.
    
    Args:
        model: Parameter.
        class_indices: Parameter.
        lookup_label_dict: Parameter.
        select_count: Parameter.
        labels: Parameter.
        save_name: Parameter.
    
    Returns:
        None: Return value.
    """
    random.seed(2) #to select the same examples on every run
    # diff 17
    # sim 2

    select_files = []
    for index in class_indices:
        select_files.extend(select_random_files_from_list(lookup_label_dict[index], select_count))

    utils.write_list_to_json(select_files, save_name[:save_name.rindex('/')],
                                           f'{save_name[save_name.rindex("/")+1:save_name.rindex(".")]}_files.json', 'files')

    select_ds = tf.data.Dataset.from_tensor_slices((select_files))
    select_ds = select_ds.map(process_video_only, num_parallel_calls=tf.data.AUTOTUNE)
    select_ds = select_ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    predictions_selection = model.predict(select_ds)
    sim_vals = cosine_similarity(predictions_selection)

    np_file_name = os.path.join(save_name[:save_name.rindex('/')], f'{save_name[save_name.rindex("/")+1:save_name.rindex(".")]}_simvals.txt')
    np.savetxt(np_file_name, sim_vals)

    all_labels = []
    for label in labels:
        for i in range(select_count):
            index_label = f'{label} {i+1}'
            all_labels.append(index_label)

    # Create the heatmap
    plt.clf()
    plt.figure(figsize=(10, 8))
    plt.imshow(sim_vals, cmap='hot', interpolation='nearest')
    plt.colorbar()

    # Set tick labels for axes
    plt.xticks(ticks=np.arange(len(all_labels)), labels=all_labels, rotation=90, fontsize=10)
    plt.yticks(ticks=np.arange(len(all_labels)), labels=all_labels, fontsize=10)

    # Formatting
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_name, dpi=300, bbox_inches='tight')

def main():
    """Main.
    
    Returns:
        None: Return value.
    """
    random.seed(444)

    global IMAGE_SIZE
    if GRAYSCALE:
        IMAGE_SIZE = (IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

    global TOTAL_FRAMES
    if MIN_FRAMES%SKIP_FRAMES>0:
        TOTAL_FRAMES+=1

    # return

    # CHANGE RUN SETTINGS HERE:
    backbone = BACKBONES[1]
    train = True
    runs = 1
    use_imagenet_weights = True
    merged = True

    if len(sys.argv) > 1:
        backbone = int(sys.argv[1])

    second_labels_file = 'UNITY-Classification-A-DF-labels.txt'
    #The first set of labels are from the folder names and the folder names are the classes.
    #The second set of labels are saved in a text file with filepaths. The classes are extracted from the filepath.
    #The first set has 17 classes and the second set has 14 classes but the files are the same. The first set is just
    #more fine-grained than the second set. The applicable classes of the first set are merged so that it matches
    #the second set of labels.
    data_folder = r'Data/Unity-Classification-A-Frames'
    save_folder = f'video_class_supcon'
    if GRAYSCALE:
        save_folder+='_gray'

    save_folder += f'_{MIN_FRAMES}min_{backbone}_epochs{EPOCHS}_batch{BATCH_SIZE}_pervid{TOTAL_FRAMES}_imagenet{use_imagenet_weights}_augroll_temp'

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # run once to extract frames from videos and save to disk

    # # random.seed(444)
    #  class_count, class_lookup,

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

    #TODO: REMOVE LATER
    train_files.extend(val_files)
    train_labels.extend(val_labels)

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

    run_losses, run_epochs = [], []
    run_val_losses, run_val_epochs = [], []
    training_times = []

    gpus = ["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4", "/gpu:5"]

    model = None
    training_info = {}
    if train:
        # train from scratch or fine-tune happens here
        model, history, ellapsed_time = train_model(train_ds=train_ds, val_ds=val_ds,
                                     output_folder=save_folder,
                                     backbone=backbone, run_index=i,
                                     num_train=len(train_files),
                                     use_imagenet_weights=use_imagenet_weights)

        loss = np.array(history.history['loss']).astype(float)
        run_losses.append(np.min(loss))
        run_epochs.append(int(np.argmin(loss)) + 1)  # 0-based index but epoch is 1-based
        training_times.append(ellapsed_time)

        training_info['run_losses'] = run_losses
        training_info['run_epochs'] = run_epochs
        training_info['training_times'] = training_times
        utils.write_dict_to_json(training_info, save_folder, f'training_info.json')

        tf.keras.backend.clear_session()

   
    print('Completed.')

if __name__ == "__main__":
    main()
