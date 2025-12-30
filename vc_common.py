"""
Common utilities shared across training / evaluation scripts.

This module removes copy/paste duplication from the original prototype scripts.
The implementations are intentionally kept identical in logic to the originals.
"""

from __future__ import annotations

import itertools
import math
import os
import random
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


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

