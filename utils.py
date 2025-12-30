import json

import cv2
import matplotlib.markers
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FormatStrFormatter
from matplotlib.collections import LineCollection
import os

import pylab as pl
import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np
from mask_volume import *

def pad_resize_preserve_aspect_ratio(img, size=1024):
    desired_size = size

    im = img.copy()
    old_size = im.shape[:2]  # old_size is in (height, width) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])


    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=color)

    return new_img

def plot_loss_per_epoch_skip(history, save_file_path, skip):
    plt.clf()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = [i for i in range(skip, len(loss))]
    plt.plot(epochs, loss[skip:], '-*', label='Training loss')
    plt.plot(epochs, val_loss[skip:], '-+', label='Val loss', color='r')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_file_path)
    plt.show()
    plt.clf()

def plot_loss_per_epoch_skip_np(loss, val_loss, save_file_path, skip):
    plt.clf()
    epochs = [i for i in range(skip, len(loss))]
    plt.plot(epochs, loss[skip:], '-*', label='Training loss')
    plt.plot(epochs, val_loss[skip:], '-+', label='Val loss', color='r')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_file_path)
    plt.show()
    plt.clf()

def plot_model(model, model_plot_name, show_summary = True):
    if show_summary:
        model.summary()
    tf.keras.utils.plot_model(model, to_file=model_plot_name, show_shapes = True, show_dtype = True)

def display_training_data(exp_folder, training_dataset):
    image_batch, label_batch = next(iter(training_dataset))

    print(image_batch.shape)

    image_batch = image_batch.numpy()
    label_batch = tf.dtypes.cast(label_batch, tf.uint8)
    label_batch = label_batch.numpy()


    for image_idx in range(image_batch.shape[0]):

        image = image_batch[image_idx]
        masks = label_batch[image_idx]

        plt.figure(figsize=(10, 10), dpi=100)
        i = 0
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image)

        for i in range(1, masks.shape[2] + 1):
            idx = i - 1
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(masks[:, :, idx])


        plt.savefig(os.path.join(exp_folder, 'images_and_label_examples.png'))
        plt.clf()

        break


def write_comparisons():
    compare_methods()

    compare_batches(1900)

    compare_enc_vs_full()

    compare_unlabelled_sets()



def write_comparisons_camus():
    compare_methods_camus()

def write_comparisons_echonet():
    compare_methods_echonet()

def compare_baseline():
    save_folder = 'Results of Baseline comparison'

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    base1 = get_baseline_scores_df('.', 'Baseline')
    base2 = get_baseline_scores_df('.', 'Baseline_444')
    base3 = get_baseline_scores_df('.', 'Baseline_444_1')  #without post-processing
    base4 = get_baseline_scores_df('.', 'Baseline_444_DiceLoss')
    base5 = get_baseline_scores_df('.', 'Baseline_444_DiceLoss1')   #without post-processing

    percentages_list = [1, 2, 3, 4, 5, 10, 15, 25, 100]
    mask = base1['percentage_train'].isin(percentages_list)
    base1 = base1[mask]
    mask = base2['percentage_train'].isin(percentages_list)
    base2 = base2[mask]
    mask = base3['percentage_train'].isin(percentages_list)
    base3 = base3[mask]
    mask = base4['percentage_train'].isin(percentages_list)
    base4 = base4[mask]
    mask = base5['percentage_train'].isin(percentages_list)
    base5 = base5[mask]

    dfs_dict = {}
    dfs_dict['old_base_bce'] = base1
    dfs_dict['base_bce'] = base2
    dfs_dict['base_bce_nopost'] = base3
    dfs_dict['base_dice'] = base4
    dfs_dict['base_dice_nopost'] = base5

    colours_dict = {
        'old_base_bce': 'blue',
        'base_bce': 'green',
        'base_bce_nopost': 'violet',
        'base_dice': 'slategray',
        'base_dice_nopost': 'red'
    }

    skip_num = 1

    # Write dice graph for test set
    plot_graph_for_downstream_scores(save_folder=save_folder, df_baseline=base1, enc_df_dict=dfs_dict,
                                     x_col_name='x_label', y_col_name='dice_test', y_label='Dice',
                                     title=f'Test Set Results',
                                     file_name=f'dice_results_test.png',
                                     skip=skip_num,
                                     max_line=True,
                                     set_upper_lim=False, set_lower_lim=True,
                                     show_baseline=False,
                                     colours_dict=colours_dict,
                                     linestyle_dict=None,
                                     add_zoom=False, add_x_axis_breaks=True)

    plot_graph_for_downstream_scores(save_folder=save_folder, df_baseline=base1, enc_df_dict=dfs_dict,
                                     x_col_name='x_label', y_col_name='dice_con', y_label='Dice',
                                     title=f'Consensus Set Results',
                                     file_name=f'dice_results_consensus.png',
                                     skip=skip_num,
                                     max_line=True,
                                     set_upper_lim=False, set_lower_lim=True,
                                     show_baseline=False,
                                     colours_dict=colours_dict,
                                     linestyle_dict=None,
                                     add_zoom=False, add_x_axis_breaks=True)

    plot_graph_for_downstream_scores(save_folder=save_folder, df_baseline=base1, enc_df_dict=dfs_dict,
                                     x_col_name='x_label', y_col_name='dice_con_top100', y_label='Dice',
                                     title=f'Consensus Set Results - Top 100',
                                     file_name=f'dice_results_consensus_top100.png',
                                     skip=skip_num,
                                     max_line=True,
                                     set_upper_lim=False, set_lower_lim=True,
                                     show_baseline=False,
                                     colours_dict=colours_dict,
                                     linestyle_dict=None,
                                     add_zoom=False, add_x_axis_breaks=True)


def compare_diff_unlabelled_sets():
    save_folder = 'Results of Rotation for Diff Unlabelled Sets'

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    baseline = get_baseline_scores_df('.', 'Baseline_444_DiceLoss')

    dtype_dict = {
        'percentage_unlabelled': 'int64',
        'num_unlabelled': 'int64',
        'percentage_train': 'float64',
        'train': 'int64',
        'dice_test': 'float64',
        'dice_test_std': 'float64',
        'dice_con': 'float64',
        'dice_con_std': 'float64',
        'dice_con_top100': 'float64',
        'dice_con_std_top100': 'float64',
        'hd_test': 'float64',
        'hd_test_std': 'float64',
        'hd_con': 'float64',
        'hd_con_std': 'float64',
        'hd_con_top100': 'float64',
        'hd_con_std_top100': 'float64',
        'val': 'int64',
        'test': 'int64',
        'consensus': 'int64',
        'x_label': 'str',
        'hue_label': 'str',
        'ef_error': 'float64',
        'ef_error_std': 'float64',
        'ed_vol_error': 'float64',
        'ed_vol_error_std': 'float64',
        'es_vol_error': 'float64',
        'es_vol_error_std': 'float64'
    }

    df1 = pd.read_csv('run_ssl_rotation_COPY_Unity_unet_dice_False_batch16/results_on_enc_1900.csv', dtype=dtype_dict)
    df2 = pd.read_csv('run_ssl_rotation_COPY_UnityALL_unet_dice_False_batch16/results_on_enc_60000.csv', dtype=dtype_dict)
    df3 = pd.read_csv('run_ssl_rotation_COPY_A4CH_unet_dice_False_batch16/results_on_enc_30000.csv', dtype=dtype_dict)

    percentages_list = [1, 2, 3, 4, 5, 10, 15, 25, 100]
    mask = baseline['percentage_train'].isin(percentages_list)
    baseline = baseline[mask]
    mask = df1['percentage_train'].isin(percentages_list)
    df1 = df1[mask]
    mask = df2['percentage_train'].isin(percentages_list)
    df2 = df2[mask]
    mask = df3['percentage_train'].isin(percentages_list)
    df3 = df3[mask]

    dfs_dict = {}
    dfs_dict['rotation_Unity_1900'] = df1
    dfs_dict['rotation_UnityALLFrames_60000'] = df2
    dfs_dict['rotation_classA_30000'] = df3

    colours_dict = {
        'rotation_Unity_1900': 'red',
        'rotation_UnityALLFrames_60000': 'green',
        'rotation_classA_30000': 'blue'
    }

    skip_num = 0

    # Write dice graph for test set
    plot_graph_for_downstream_scores(save_folder=save_folder, df_baseline=baseline, enc_df_dict=dfs_dict,
                                     x_col_name='x_label', y_col_name='dice_test', y_label='Dice',
                                     title=f'Test Set Results',
                                     file_name=f'dice_results_test.png',
                                     skip=skip_num,
                                     max_line=True,
                                     set_upper_lim=False, set_lower_lim=True,
                                     show_baseline=True,
                                     colours_dict=colours_dict,
                                     linestyle_dict=None,
                                     add_zoom=False, add_x_axis_breaks=True)

    plot_graph_for_downstream_scores(save_folder=save_folder, df_baseline=baseline, enc_df_dict=dfs_dict,
                                     x_col_name='x_label', y_col_name='dice_con', y_label='Dice',
                                     title=f'Consensus Set Results',
                                     file_name=f'dice_results_consensus.png',
                                     skip=skip_num,
                                     max_line=True,
                                     set_upper_lim=False, set_lower_lim=True,
                                     show_baseline=True,
                                     colours_dict=colours_dict,
                                     linestyle_dict=None,
                                     add_zoom=False, add_x_axis_breaks=True)

    plot_graph_for_downstream_scores(save_folder=save_folder, df_baseline=baseline, enc_df_dict=dfs_dict,
                                     x_col_name='x_label', y_col_name='dice_con_top100', y_label='Dice',
                                     title=f'Consensus Set Results - Top 100',
                                     file_name=f'dice_results_consensus_top100.png',
                                     skip=skip_num,
                                     max_line=True,
                                     set_upper_lim=False, set_lower_lim=True,
                                     show_baseline=True,
                                     colours_dict=colours_dict,
                                     linestyle_dict=None,
                                     add_zoom=False, add_x_axis_breaks=True)


def compare_temp():
    save_folder = 'Results of TEMP'

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    baseline = get_baseline_scores_df('.', 'Baseline_444_1')

    dtype_dict = {
        'percentage_unlabelled': 'int64',
        'num_unlabelled': 'int64',
        'percentage_train': 'float64',
        'train': 'int64',
        'dice_test': 'float64',
        'dice_test_std': 'float64',
        'dice_con': 'float64',
        'dice_con_std': 'float64',
        'dice_con_top100': 'float64',
        'dice_con_std_top100': 'float64',
        'hd_test': 'float64',
        'hd_test_std': 'float64',
        'hd_con': 'float64',
        'hd_con_std': 'float64',
        'hd_con_top100': 'float64',
        'hd_con_std_top100': 'float64',
        'val': 'int64',
        'test': 'int64',
        'consensus': 'int64',
        'x_label': 'str',
        'hue_label': 'str',
        'ef_error': 'float64',
        'ef_error_std': 'float64',
        'ed_vol_error': 'float64',
        'ed_vol_error_std': 'float64',
        'es_vol_error': 'float64',
        'es_vol_error_std': 'float64'
    }

    df1 = pd.read_csv('run_SimCLR_Echo_Unity_unet-enc_bce_False_batch32/results_on_enc_1900.csv', dtype=dtype_dict)
    df2 = pd.read_csv('run_Patch_rand_unity_unet_bce_False_batch16/results_on_enc_1900.csv', dtype=dtype_dict)

    percentages_list = [1, 2, 3, 4, 5, 10, 15, 25, 100]
    mask = baseline['percentage_train'].isin(percentages_list)
    baseline = baseline[mask]

    mask = df1['percentage_train'].isin(percentages_list)
    df1 = df1[mask]

    mask = df2['percentage_train'].isin(percentages_list)
    df2 = df2[mask]

    dfs_dict = {}
    dfs_dict['SimCLR'] = df1
    dfs_dict['Region-Mask'] = df2

    colours_dict = {'SimCLR': 'red', 'Region-Mask': 'green'}

    skip_num = 0

    # Write dice graph for test set
    plot_graph_for_downstream_scores(save_folder=save_folder, df_baseline=baseline, enc_df_dict=dfs_dict,
                                     x_col_name='x_label', y_col_name='dice_test', y_label='Dice',
                                     title=f'Test Set Results',
                                     file_name=f'dice_results_test.png',
                                     skip=skip_num,
                                     max_line=True,
                                     set_upper_lim=False, set_lower_lim=True,
                                     show_baseline=True,
                                     colours_dict=colours_dict,
                                     linestyle_dict=None,
                                     add_zoom=False, add_x_axis_breaks=True)

    plot_graph_for_downstream_scores(save_folder=save_folder, df_baseline=baseline, enc_df_dict=dfs_dict,
                                     x_col_name='x_label', y_col_name='dice_con', y_label='Dice',
                                     title=f'Consensus Set Results',
                                     file_name=f'dice_results_consensus.png',
                                     skip=skip_num,
                                     max_line=True,
                                     set_upper_lim=False, set_lower_lim=True,
                                     show_baseline=True,
                                     colours_dict=colours_dict,
                                     linestyle_dict=None,
                                     add_zoom=False, add_x_axis_breaks=True)

    plot_graph_for_downstream_scores(save_folder=save_folder, df_baseline=baseline, enc_df_dict=dfs_dict,
                                     x_col_name='x_label', y_col_name='dice_con_top100', y_label='Dice',
                                     title=f'Consensus Set Results - Top 100',
                                     file_name=f'dice_results_consensus_top100.png',
                                     skip=skip_num,
                                     max_line=True,
                                     set_upper_lim=False, set_lower_lim=True,
                                     show_baseline=True,
                                     colours_dict=colours_dict,
                                     linestyle_dict=None,
                                     add_zoom=False, add_x_axis_breaks=True)


def compare_leastrep():
    save_folder = 'Results of Different Methods Compared leastrep'

    main_folders = [

        'run_Patch_rand_leastrep_unet_bce_False_batch16',

        'run_SimCLR_Echo_leastrep_unet_bce_True_batch20',
        # 'run_Novel_2D_New_unet_bce_mixed_batch64',
        # 'run_split_unet_bce_True_batch20',
        # 'run_MTL_unet_bce_True_batch20',
    ]
    names = [
        #'Barlow Twins',
        'Focal-mask',
        #'Rotation',
        #'SimSiam',
        'SimCLR',

        # 'Mixed',

        # 'Split',
        # 'MTL',
    ]

    #                 'Barlow Twins': 'blue',
    #                 'Rotation': 'mediumseagreen',
    #                 #'Mixed': 'magenta',
    #                 'Focal-mask': 'orchid',
    #                 'SimSiam': 'darkorange',
    #                 #'Split': 'red',
    #                 #'MTL': 'yellow',

    colours_dict = {
        'Barlow Twins': 'cornflowerblue',
        'Focal-mask': 'salmon',
        'Rotation': 'mediumseagreen',
        # 'Mixed': '#FCDE9C',
        'SimSiam': 'orchid',
        'SimCLR': 'orange',

        # 'Split': 'red',
        # 'MTL': 'yellow',
    }

    write_results_for_methods_comparison(save_folder, main_folders, names, 30000, True, colours_dict,
                                         linestyle_dict=None, bar_width=0.14)

def compare_patches():
    save_folder = 'Results of Masks'

    main_folders = [  # 'run_Patch_unet_bce_False_batch16',
        'run_Patch_unet_bce_True_batch20',
        # 'run_Patch_unet_bce_True_batch16',
        'run_ssl_rotation_unet_bce_False_batch16',
    ]
    names = [  # 'Focal-mask0',
        'Focal-mask1',
        # 'Focal-mask2',
        'Rotation',
    ]

    colours_dict = {  # 'Focal-mask0': 'pink',
        'Focal-mask1': 'orchid',
        # 'Focal-mask2': 'blue',
        'Rotation': 'mediumseagreen',
    }

    write_results_for_methods_comparison(save_folder, main_folders, names, 30000, True, colours_dict)


def compare_methods():
    save_folder = 'Results of Different Methods Compared'

    main_folders = [
                    'run_BTwin_unet_bce_True_batch20',
                    'run_Patch_rand_unity_patch_unet_bce_False_batch16',
                    'run_Patch_rand_unity_horizontal_unet_bce_False_batch16',
                    'run_ssl_rotation_unet_bce_False_batch16',
                    'run_SimCLR_Unity_unet_bce_True_batch20',
                    'run_Split2_unet_bce_False_batch16',
                    ]
    names = [
        'Barlow Twins',
        'Region-Based',#'Masking-A',
        'Strip-Based',#'Masking-B',
        'Rotation',
        'SimCLR',
        'Split',
             ]

    #     'BTwins',
    #     'Region',  # 'Masking-A',
    #     'Strip',  # 'Masking-B',
    #     'Rotation',
    #     'SimCLR',
    #     'Split',

    #                 'Barlow Twins': 'blue',
    #                 'Rotation': 'mediumseagreen',
    #                 #'Mixed': 'magenta',
    #                 'Focal-mask': 'orchid',
    #                 'SimSiam': 'darkorange',
    #                 #'Split': 'red',
    #                 #'MTL': 'yellow',

    colours_dict = {
                    'Barlow Twins': 'cornflowerblue',
                    'Region-Based': 'salmon',
                    'Strip-Based': 'orchid',
                    'Rotation': 'mediumseagreen',
                    'SimCLR': 'orange',
                    'Split': 'yellowgreen',
                    'Rand': 'slategray'
                    # 'MTL': 'yellow',
                    }


    write_results_for_methods_comparison_new_plots(save_folder, main_folders, names, 1900, True, colours_dict, linestyle_dict=None, baseline_folder = 'Baseline_444_Final')


def compare_methods_camus():
    save_folder = 'Results of Different Methods Compared Camus'

    main_folders = [
                    'run_BTwin_camus_unet_bce_True_batch20',
                    'run_Patch_rand_camus_patch_unet_bce_False_batch16',
                    'run_Patch_rand_camus_strip_unet_bce_False_batch16',
                    'run_ssl_rotation_camus_unet_bce_False_batch16',
                    'run_SimCLR_camus_unet_bce_True_batch20',
                    'run_Split2_camus_unet_bce_False_batch16',
                    ]
    names = [
        'Barlow Twins',
        'Region-Based',#'Masking-A',
        'Strip-Based',#'Masking-B',
        'Rotation',
        'SimCLR',
        'Split',
             ]

    colours_dict = {
                    'Barlow Twins': 'cornflowerblue',
                    'Region-Based': 'salmon',
                    'Strip-Based': 'orchid',
                    'Rotation': 'mediumseagreen',
                    'SimCLR': 'orange',
                    'Split': 'yellowgreen',
                    'Rand': 'slategray'
                    # 'MTL': 'yellow',
                    }


    write_results_for_methods_comparison_new_plots_test_only(save_folder, main_folders, names, 1900, True, colours_dict, linestyle_dict=None, baseline_folder = 'Baseline_444_Final_camus')

def compare_methods_echonet():
    save_folder = 'Results of Different Methods Compared Echonet'

    main_folders = [
                    'run_BTwin_Echonet_unet_bce_True_batch20',
                    'run_Patch_rand_Echonet_patch_unet_bce_False_batch16',
                    'run_Patch_rand_Echonet_horizontal_unet_bce_False_batch16',
                    'run_ssl_rotation_echonet_unet_bce_False_batch16',
                    'run_SimCLR_Echonet_unet_bce_True_batch20',
                    'run_Split2_Echonet_unet_bce_False_batch16',
                    ]
    names = [
        'Barlow Twins',
        'Region-Based',#'Masking-A',
        'Strip-Based',#'Masking-B',
        'Rotation',
        'SimCLR',
        'Split',
             ]

    colours_dict = {
                    'Barlow Twins': 'cornflowerblue',
                    'Region-Based': 'salmon',
                    'Strip-Based': 'orchid',
                    'Rotation': 'mediumseagreen',
                    'SimCLR': 'orange',
                    'Split': 'yellowgreen',
                    'Rand': 'slategray'
                    # 'MTL': 'yellow',
                    }


    write_results_for_methods_comparison_new_plots(save_folder, main_folders, names, 1900, True, colours_dict, linestyle_dict=None, baseline_folder = 'Baseline_444_Final_Echonet')


def compare_batches(pretraining_size):
    #
    #
    #
    #
    #

    ###################################################################

    save_folder = f'Results Batch Sizes SimCLR'

    main_folders = [f'run_SimCLR_Unity_unet_bce_True_batch{8}',
                    f'run_SimCLR_Unity_unet_bce_True_batch{16}',
                    f'run_SimCLR_Unity_unet_bce_True_batch{20}',
                    ]
    names = [f'SimCLR-{8}',
             f'SimCLR-{16}',
             f'SimCLR-{20}'
             ]


    colours_dict = {f'SimCLR-{8}': 'tomato',
                    f'SimCLR-{16}': 'peru',
                    f'SimCLR-{20}': 'orange',
                    f'Rand': 'slategray'}

    linestyle_dict = {f'SimCLR-{8}': (0, (1, 1)),
                      f'SimCLR-{16}': (0, (3, 1, 1, 1, 1, 1)),
                      f'SimCLR-{20}': 'dashed'}

    write_results_for_methods_comparison(save_folder, main_folders, names, pretraining_size, True, colours_dict, linestyle_dict)


def compare_enc_vs_full():
    # encoder vs full unet:
    save_folder = 'Results Enc vs Full'

    main_folders = [f'run_SimCLR_Unity_unet-enc_bce_True_batch{64}',
                    f'run_SimCLR_Unity_unet-enc_bce_True_batch{20}',
                    f'run_SimCLR_Unity_unet_bce_True_batch{20}',
                    ]
    names = [f'SimCLR-enc-{64}',
             f'SimCLR-enc-{20}',
             f'SimCLR-{20}'
             ]

    colours_dict = {f'SimCLR-enc-{64}': 'tomato',
                    f'SimCLR-enc-{20}': 'peru',
                    f'SimCLR-{20}': 'orange',
                    'Rand': 'slategray'}

    linestyle_dict = {f'SimCLR-enc-{64}': (0, (1, 1)),
                      f'SimCLR-enc-{20}': (0, (3, 1, 1, 1, 1, 1)),
                      f'SimCLR-{20}': 'dashed'}

    #                                      linestyle_dict)
    write_results_for_methods_comparison_new_plots(save_folder, main_folders, names, 1900, True, colours_dict,
                                                   linestyle_dict=linestyle_dict, baseline_folder='Baseline_444_Final')

def compare_unlabelled_sets():
    # encoder vs full unet:
    save_folder = 'Results unlabelled sets'

    main_folders = [f'run_SimCLR_UnityFrames_unet_bce_True_batch{20}',
                    f'run_SimCLR_Unity_unet_bce_True_batch{20}',
                    f'run_SimCLR_A4CHLV_unet_bce_True_batch{20}',
                    f'run_SimCLR_A4CHLV_1900_unet_bce_True_batch{20}',
                    ]
    names = [
            f'Unlabelled-A (60K)',
            f'Unlabelled-A (2057)',
             f'Unlabelled-B (100K)',
             f'Unlabelled-B (2057)',
             ]

    colours_dict = {f'Unlabelled-B (100K)': 'crimson',
                    f'Unlabelled-B (2057)': 'darkturquoise',
                    f'Unlabelled-A (60K)': 'sienna',
                    f'Unlabelled-A (2057)': 'orange',
                    'Rand': 'slategray'}

    linestyle_dict = {f'Unlabelled-B (100K)': (0, (3, 1, 1, 1, 1, 1)),
                      f'Unlabelled-B (2057)': (0, (3, 1, 1, 1, 1, 1)),
                      f'Unlabelled-A (60K)': 'dashed',
                      f'Unlabelled-A (2057)': 'dashed'}

    #                                      linestyle_dict)
    write_results_for_methods_comparison_new_plots(save_folder, main_folders, names, 1900, True, colours_dict,
                                                   linestyle_dict=linestyle_dict, baseline_folder='Baseline_444_Final')


def get_baseline_scores_df(scores_folder, baseline_folder='Baseline_444_Final'):
    df_base = pd.DataFrame(
        columns=['percentage_unlabelled', 'num_unlabelled',  # not used, just for combining dataframes later
                 'percentage_train', 'train',
                 'dice_test', 'dice_test_std',
                 'dice_con', 'dice_con_std',
                 'dice_con_top100', 'dice_con_std_top100',
                 'hd_test', 'hd_test_std',
                 'hd_con', 'hd_con_std',
                 'hd_con_top100', 'hd_con_std_top100',
                 'iou_test', 'iou_test_std',
                 'iou_con', 'iou_con_std',
                 'iou_con_top100', 'iou_con_std_top100',
                 'val', 'test', 'consensus',
                 'x_label', 'hue_label',
                 'ef_error', 'ef_error_std',
                 'ed_vol_error', 'ed_vol_error_std',
                 'es_vol_error', 'es_vol_error_std'])

    df_base['percentage_unlabelled'] = df_base['percentage_unlabelled'].astype(int)
    df_base['num_unlabelled'] = df_base['num_unlabelled'].astype(int)
    df_base['percentage_train'] = df_base['percentage_train'].astype(int)
    df_base['train'] = df_base['train'].astype(int)
    df_base['dice_test'] = df_base['dice_test'].astype(float)
    df_base['dice_test_std'] = df_base['dice_test_std'].astype(float)
    df_base['dice_con'] = df_base['dice_con'].astype(float)
    df_base['dice_con_std'] = df_base['dice_con_std'].astype(float)
    df_base['dice_con_top100'] = df_base['dice_con_top100'].astype(float)
    df_base['dice_con_std_top100'] = df_base['dice_con_std_top100'].astype(float)
    df_base['hd_test'] = df_base['hd_test'].astype(float)
    df_base['hd_test_std'] = df_base['hd_test_std'].astype(float)
    df_base['hd_con'] = df_base['hd_con'].astype(float)
    df_base['hd_con_std'] = df_base['hd_con_std'].astype(float)
    df_base['hd_con_top100'] = df_base['hd_con_top100'].astype(float)
    df_base['hd_con_std_top100'] = df_base['hd_con_std_top100'].astype(float)

    df_base['iou_test'] = df_base['iou_test'].astype(float)
    df_base['iou_test_std'] = df_base['iou_test_std'].astype(float)
    df_base['iou_con'] = df_base['iou_con'].astype(float)
    df_base['iou_con_std'] = df_base['iou_con_std'].astype(float)
    df_base['iou_con_top100'] = df_base['iou_con_top100'].astype(float)
    df_base['iou_con_std_top100'] = df_base['iou_con_std_top100'].astype(float)

    df_base['val'] = df_base['val'].astype(int)
    df_base['test'] = df_base['test'].astype(int)
    df_base['consensus'] = df_base['consensus'].astype(int)
    df_base['x_label'] = df_base['x_label'].astype(str)
    df_base['hue_label'] = df_base['hue_label'].astype(str)
    df_base['ef_error'] = df_base['ef_error'].astype(float)
    df_base['ef_error_std'] = df_base['ef_error_std'].astype(float)
    df_base['ed_vol_error'] = df_base['ed_vol_error'].astype(float)
    df_base['ed_vol_error_std'] = df_base['ed_vol_error_std'].astype(float)
    df_base['es_vol_error'] = df_base['es_vol_error'].astype(float)
    df_base['es_vol_error_std'] = df_base['es_vol_error_std'].astype(float)

    baseline_folder = os.path.join(scores_folder, baseline_folder)
    files = os.listdir(baseline_folder)
    baseline_dices_test, baseline_dices_consensus = [], []
    baseline_hds_test, baseline_hds_consensus = [], []
    for file in files:
        if not 'baseline_results' in file:
            continue

        res_dict = load_json(os.path.join(baseline_folder, file))
        dice_test = res_dict['run_dices_test']
        dice_test_avg = np.mean(np.array(dice_test).astype(float))
        dice_test_std = np.std(np.array(dice_test).astype(float))
        hd_test = res_dict['run_hds_test']
        hd_test_avg = np.mean(np.array(hd_test).astype(float))
        hd_test_std = np.std(np.array(hd_test).astype(float))
        iou_test = res_dict['run_iou_test']
        iou_test_avg = np.mean(np.array(iou_test).astype(float))
        iou_test_std = np.std(np.array(iou_test).astype(float))
        dice_con = res_dict['run_dices_consensus']
        dice_con_avg = np.mean(np.array(dice_con).astype(float))
        dice_con_std = np.std(np.array(dice_con).astype(float))
        hd_con = res_dict['run_hds_consensus']
        hd_con_avg = np.mean(np.array(hd_con).astype(float))
        hd_con_std = np.std(np.array(hd_con).astype(float))
        iou_con = res_dict['run_iou_consensus']
        iou_con_avg = np.mean(np.array(iou_con).astype(float))
        iou_con_std = np.std(np.array(iou_con).astype(float))
        dice_con_top100 = res_dict['run_dices_consensus_top100']
        dice_con_avg_top100 = np.mean(np.array(dice_con_top100).astype(float))
        dice_con_std_top100 = np.std(np.array(dice_con_top100).astype(float))
        hd_con_top100 = res_dict['run_hds_consensus_top100']
        hd_con_avg_top100 = np.mean(np.array(hd_con_top100).astype(float))
        hd_con_std_top100 = np.std(np.array(hd_con_top100).astype(float))
        iou_con_top100 = res_dict['run_iou_consensus_top100']
        iou_con_avg_top100 = np.mean(np.array(iou_con_top100).astype(float))
        iou_con_std_top100 = np.std(np.array(iou_con_top100).astype(float))
        num_train = int(res_dict['num_train'])
        percentage_train = res_dict['percentage_train']
        if percentage_train is None: percentage_train = 100
        num_val = int(res_dict['num_val'])
        num_test = int(res_dict['num_test'])
        num_consensus = float(res_dict['num_consensus'])
        x_label = str(percentage_train) #f'{num_train}({percentage_train}%)'
        hue_label = f'Supervised(random initialisation)'
        ef_errors = np.array(res_dict['avg_ef_errors_per_run'])
        ed_errors = np.array(res_dict['avg_ed_vol_errors_per_run'])
        es_errors = np.array(res_dict['avg_es_vol_errors_per_run'])

        row = {'percentage_unlabelled': 0, 'num_unlabelled': 0,
               'percentage_train': percentage_train, 'train': num_train,
               'dice_test': dice_test_avg, 'dice_test_std': dice_test_std,
               'dice_con': dice_con_avg, 'dice_con_std': dice_con_std,
               'dice_con_top100': dice_con_avg_top100, 'dice_con_std_top100': dice_con_std_top100,
               'hd_test': hd_test_avg, 'hd_test_std': hd_test_std,
               'hd_con': hd_con_avg, 'hd_con_std': hd_con_std,
               'hd_con_top100': hd_con_avg_top100, 'hd_con_std_top100': hd_con_std_top100,
               'iou_test': iou_test_avg, 'iou_test_std': iou_test_std,
               'iou_con': iou_con_avg, 'iou_con_std': iou_con_std,
               'iou_con_top100': iou_con_avg_top100, 'iou_con_std_top100': iou_con_std_top100,
               'val': num_val, 'test': num_test, 'consensus': num_consensus,
               'x_label': x_label, 'hue_label': hue_label,
               'ef_error': np.mean(ef_errors), 'ef_error_std': np.std(ef_errors),
               'ed_vol_error': np.mean(ed_errors), 'ed_vol_error_std': np.std(ed_errors),
               'es_vol_error': np.mean(es_errors), 'es_vol_error_std': np.std(es_errors)
               }
        df_base = df_base.append(row, ignore_index=True)

        baseline_dices_test.append(dice_test_avg)
        baseline_dices_consensus.append(dice_con_avg)
        baseline_hds_test.append(hd_test_avg)
        baseline_hds_consensus.append(hd_con_avg)

    if len(baseline_dices_test) == 0:
        return

    # save df as csv
    df_base.sort_values(by=['percentage_train'], inplace=True)

    return df_base



def axis_break(axis, xpos=[0.1, 0.125], slant=1.5):
    d = slant  # proportion of vertical to horizontal extent of the slanted line
    anchor = (xpos[0], -0.05)
    w = xpos[1] - xpos[0]
    h = 0.1

    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=20, zorder=3,
                linestyle="none", color='k', mec='k', mew=3, clip_on=False)
    axis.add_patch(patches.Rectangle(
        anchor, w, h, fill=True, color="white",
        transform=axis.transAxes, clip_on=False, zorder=3)
    )
    axis.plot(xpos, [0, 0], transform=axis.transAxes, **kwargs)



def export_legend(legend, filename="legend.png"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def plot_graph_for_downstream_scores_combined(save_folder,
                                              df_baseline, enc_df_dict,
                                              colours_dict = None, linestyle_dict=None,
                                              ):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os

    data = {}
    data["% (No.)"] = ["1 (20)", "2 (41)", "3 (61)", "4 (82)", "5 (102)", "10 (205)", "15 (308)", "25 (514)",
                       "100 (2057)"]

    data["Rand Test"] = df_baseline['dice_test'].to_numpy()
    for key, val in enc_df_dict.items():
        data[key + ' Test'] = val['dice_test'].to_numpy()

    data["Rand Consensus"] = df_baseline['dice_con'].to_numpy()
    for key, val in enc_df_dict.items():
        data[key + ' Consensus'] = val['dice_con'].to_numpy()

    data["Rand T100"] = df_baseline['dice_con_top100'].to_numpy()
    for key, val in enc_df_dict.items():
        data[key + ' T100'] = val['dice_con_top100'].to_numpy()

    # Create DataFrame
    df = pd.DataFrame(data)

    # Convert percentage column to integers for plotting
    df["% (No.)"] = df["% (No.)"].str.extract('(\d+)').astype(int)

    # Extract relevant data for each dataset
    ln = len(enc_df_dict)
    test_data = df.iloc[:, 1:ln + 2]
    consensus_data = df.iloc[:, ln + 2:ln + 2 + ln + 1]
    t100_data = df.iloc[:, ln + 2 + ln + 1:]

    datasets = [test_data, consensus_data, t100_data]
    titles = ["Test Dataset", "Consensus Dataset", "T100 Dataset"]

    custom_positions = [1, 2, 3, 4, 5, 10, 15, 25, 100]
    custom_labels = [str(x) for x in custom_positions]

    title_size = 22
    label_size = 20
    tick_size = 18
    legend_font_size = 14

    share_y = True
    figsize = (20, 11) if share_y else (20, 7)

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=share_y, dpi=300)
    data_min = 9999999
    data_max = -1

    for i, (ax, data, title) in enumerate(zip(axes, datasets, titles)):
        for col in data.columns:
            y_values = [data.loc[df["% (No.)"] == x, col].values[0] for x in custom_positions if
                        x in df["% (No.)"].values]
            label_name = col.replace(title.split()[0], "").strip()
            colour = colours_dict[label_name]

            ax.plot(custom_positions, y_values, marker='o', label=label_name, linewidth=2, markersize=6, color=colour)

        ax.set_title(title, fontsize=title_size)
        ax.set_xlabel("% Labelled Data", fontsize=label_size)

        if i == 0:
            ax.set_ylabel("Dice Score", fontsize=label_size)
        else:
            ax.set_ylabel("")

        ax.tick_params(axis='both', width=2, size=6)

        # Set x-axis to log scale
        ax.set_xscale('log')
        ax.set_xticks(custom_positions)
        ax.get_xaxis().set_major_formatter(
            plt.FuncFormatter(lambda x, _: str(int(x))))  # Convert tick labels to integers

        ax.grid(True, linestyle='--', alpha=0.7)

        dataset_min = data.min().min()
        dataset_max = data.max().max()
        ax.set_ylim(dataset_min - 0.01, dataset_max + 0.01)

        if dataset_min < data_min:
            data_min = dataset_min
        if dataset_max > data_max:
            data_max = dataset_max

        max_rand_value = data.loc[df["% (No.)"] <= 100, f"Rand {title.split()[0]}"].max()
        ax.axhline(y=max_rand_value, color='slategray', linestyle='--', linewidth=2)

        x_tick_locations = ax.get_xticks()
        ax.text(x=x_tick_locations[0], y=max_rand_value * 1.0005, weight='bold', fontsize=17,
                s=str(np.round(max_rand_value, 3)))

    if share_y:
        plt.ylim(bottom=data_min - 0.01)
        plt.yticks(np.round(np.arange(float(data_min), float(data_max + 0.01), 0.02), 2), fontsize=tick_size)

    # Make the tick lines bigger for all ticks
    plt.sca(axes[0])
    plt.yticks(fontsize=tick_size)
    plt.xticks(fontsize=tick_size)
    plt.sca(axes[1])
    plt.yticks(fontsize=tick_size)
    plt.xticks(fontsize=tick_size)
    plt.sca(axes[2])
    plt.yticks(fontsize=tick_size)
    plt.xticks(fontsize=tick_size)

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.savefig(os.path.join(save_folder, f'combined_sharey{share_y}_log_xscale_no_legend.png'), dpi=300, bbox_inches='tight')

    plt.subplots_adjust(wspace=3.5)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.94, 0.5), fontsize=legend_font_size)

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.show()

    export_legend(legend, os.path.join(save_folder, 'combined_legend_log_xscale.png'))
    plt.savefig(os.path.join(save_folder, f'combined_sharey{share_y}_log_xscale.png'), dpi=300, bbox_inches='tight')

    plt.clf()

def plot_graph_for_downstream_scores_combined_test_only(save_folder,
                                              df_baseline, enc_df_dict,
                                              colours_dict = None, linestyle_dict=None,
                                              ):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os

    data = {}
    data["% (No.)"] = ["1 (20)", "2 (41)", "3 (61)", "4 (82)", "5 (102)", "10 (205)", "15 (308)", "25 (514)",
                       "100 (2057)"]

    data["Rand Test"] = df_baseline['dice_test'].to_numpy()
    for key, val in enc_df_dict.items():
        data[key + ' Test'] = val['dice_test'].to_numpy()

    #

    # Create DataFrame
    df = pd.DataFrame(data)

    # Convert percentage column to integers for plotting
    df["% (No.)"] = df["% (No.)"].str.extract('(\d+)').astype(int)

    # Extract relevant data for each dataset
    ln = len(enc_df_dict)
    test_data = df.iloc[:, 1:ln + 2]
    consensus_data = df.iloc[:, ln + 2:ln + 2 + ln + 1]
    t100_data = df.iloc[:, ln + 2 + ln + 1:]

    datasets = [test_data]#, consensus_data, t100_data]
    titles = ["CAMUS Test Dataset"]#, "Consensus Dataset", "T100 Dataset"]

    custom_positions = [1, 2, 3, 4, 5, 10, 15, 25, 100]
    custom_labels = [str(x) for x in custom_positions]

    title_size = 22
    label_size = 20
    tick_size = 18
    legend_font_size = 14

    share_y = True
    figsize = (20, 11) if share_y else (20, 7)

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=share_y, dpi=300)
    data_min = 9999999
    data_max = -1

    for i, (ax, data, title) in enumerate(zip(axes, datasets, titles)):
        for col in data.columns:
            y_values = [data.loc[df["% (No.)"] == x, col].values[0] for x in custom_positions if
                        x in df["% (No.)"].values]
            label_name = col.replace(title.split()[1], "").strip()
            colour = colours_dict[label_name]

            ax.plot(custom_positions, y_values, marker='o', label=label_name, linewidth=2, markersize=6, color=colour)

        ax.set_title(title, fontsize=title_size)
        ax.set_xlabel("% Labelled Data", fontsize=label_size)

        if i == 0:
            ax.set_ylabel("Dice Score", fontsize=label_size)
        else:
            ax.set_ylabel("")

        ax.tick_params(axis='both', width=2, size=6)

        # Set x-axis to log scale
        ax.set_xscale('log')
        ax.set_xticks(custom_positions)
        ax.get_xaxis().set_major_formatter(
            plt.FuncFormatter(lambda x, _: str(int(x))))  # Convert tick labels to integers

        ax.grid(True, linestyle='--', alpha=0.7)

        dataset_min = data.min().min()
        dataset_max = data.max().max()
        ax.set_ylim(dataset_min - 0.01, dataset_max + 0.01)

        if dataset_min < data_min:
            data_min = dataset_min
        if dataset_max > data_max:
            data_max = dataset_max

        max_rand_value = data.loc[df["% (No.)"] <= 100, f"Rand {title.split()[1]}"].max()
        ax.axhline(y=max_rand_value, color='slategray', linestyle='--', linewidth=2)

        x_tick_locations = ax.get_xticks()
        ax.text(x=x_tick_locations[0], y=max_rand_value * 1.0005, weight='bold', fontsize=17,
                s=str(np.round(max_rand_value, 3)))

    if share_y:
        plt.ylim(bottom=data_min - 0.01)
        plt.yticks(np.round(np.arange(float(data_min), float(data_max + 0.01), 0.02), 2), fontsize=tick_size)

    # Make the tick lines bigger for all ticks
    plt.sca(axes[0])
    plt.yticks(fontsize=tick_size)
    plt.xticks(fontsize=tick_size)
    plt.sca(axes[1])
    plt.yticks(fontsize=tick_size)
    plt.xticks(fontsize=tick_size)
    plt.sca(axes[2])
    plt.yticks(fontsize=tick_size)
    plt.xticks(fontsize=tick_size)

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.savefig(os.path.join(save_folder, f'combined_sharey{share_y}_log_xscale_no_legend.png'), dpi=300, bbox_inches='tight')

    plt.subplots_adjust(wspace=3.5)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.94, 0.5), fontsize=legend_font_size)

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.show()

    export_legend(legend, os.path.join(save_folder, 'combined_legend_log_xscale.png'))
    plt.savefig(os.path.join(save_folder, f'combined_sharey{share_y}_log_xscale.png'), dpi=300, bbox_inches='tight')

    # -------- NEW CODE TO SAVE FIRST SUBFIGURE ALONE --------
    # Get position of first subfigure (in figure coordinates)
    #
    # # Create a new figure and copy the first axes
    #
    #
    # # Copy labels, title, ticks, etc.
    #
    # # Re-add horizontal line and text annotation
    #

    plt.clf()

def plot_graph_for_downstream_scores_combined1(save_folder,
                                              df_baseline, enc_df_dict,
                                              colours_dict = None, linestyle_dict=None,
                                              ):
    data = {}
    data["% (No.)"] = ["1 (20)", "2 (41)", "3 (61)", "4 (82)", "5 (102)", "10 (205)", "15 (308)", "25 (514)", "100 (2057)"]

    data["Rand Test"] =  df_baseline['dice_test'].to_numpy()
    for key, val in enc_df_dict.items():
        data[key+' Test'] = val['dice_test'].to_numpy()

    data["Rand Consensus"] = df_baseline['dice_con'].to_numpy()
    for key, val in enc_df_dict.items():
        data[key+' Consensus'] = val['dice_con'].to_numpy()

    data["Rand T100"] = df_baseline['dice_con_top100'].to_numpy()
    for key, val in enc_df_dict.items():
        data[key+' T100'] = val['dice_con_top100'].to_numpy()

    # Create DataFrame
    df = pd.DataFrame(data)

    # Convert percentage column to integers for better plotting
    df["% (No.)"] = df["% (No.)"].str.extract('(\d+)').astype(int)

    # Extract relevant data for each dataset
    ln=len(enc_df_dict)
    test_data = df.iloc[:, 1:ln+2]  # Test dataset columns
    consensus_data = df.iloc[:, ln+2:ln+2+ln+1]  # Consensus dataset columns
    t100_data = df.iloc[:, ln+2+ln+1:]  # T100 dataset columns

    datasets = [test_data, consensus_data, t100_data]
    titles = ["Test Dataset", "Consensus Dataset", "T100 Dataset"]

    # Update custom positions and labels to exclude 100%, going up to 15%
    custom_positions = [1, 2, 3, 4, 5, 10, 15]
    custom_labels = [str(x) for x in custom_positions]

    # Adjusted positions for visualization (excluding 25% and 100%)
    adjusted_positions = [1, 2, 3, 4, 5, 7, 9]

    title_size = 22
    label_size = 18  # 40
    tick_size = 18  # 36
    inset_tick_size = 34  # 34
    legend_font_size = 44  # 30

    share_y = True
    # Create subplots for each dataset with adjusted x-axis spacing and improved appearance
    figsze = (20, 7)
    if share_y:
        figsze = (20, 11)
    fig, axes = plt.subplots(1, 3, figsize=figsze, sharey=share_y, dpi=300)
    data_min = 9999999
    data_max = -1
    for i, (ax, data, title) in enumerate(zip(axes, datasets, titles)):
        for col in data.columns:
            # Extract values up to 15% only
            y_values = [data.loc[df["% (No.)"] == x, col].values[0] for x in custom_positions if
                        x in df["% (No.)"].values]
            label_name = col.replace(title.split()[0], "").strip()  # Remove dataset name, keep SSL type
            colour = colours_dict[label_name]
            # if linestyle_dict is not None:
            ax.plot(adjusted_positions, y_values, marker='o', label=label_name, linewidth=2, markersize=6, color=colour)

        ax.set_title(title, fontsize=title_size)#, fontweight='bold')
        ax.set_xlabel("% Labelled Data", fontsize=label_size)

        # Remove y-axis title for the second and third subplots
        if i == 0:
            ax.set_ylabel("Dice Score", fontsize=label_size)
        else:
            ax.set_ylabel("")

        #Make tick lines thicker
        ax.tick_params(axis='both', width=2, size=6)

        # Adjust x-axis with custom spacing and tick labels
        ax.set_xticks(adjusted_positions)
        ax.set_xticklabels(custom_labels, fontsize=tick_size)

        # Adjust size of y tick labels

        # Customize grid appearance
        ax.grid(True, linestyle='--', alpha=0.7)

        # Set individual y-axis limits with a small margin
        dataset_min = data.min().min()
        dataset_max = data.max().max()
        ax.set_ylim(dataset_min - 0.01, dataset_max + 0.01)

        if dataset_min < data_min:
            data_min = dataset_min

        if dataset_max > data_max:
            data_max = dataset_max

        # Add horizontal line for maximum Rand performance in the dataset (up to 15%)
        max_rand_value = data.loc[df["% (No.)"] <= 100, f"Rand {title.split()[0]}"].max()
        ax.axhline(y=max_rand_value, color='slategray', linestyle='--', linewidth=2)

        #Add max random score as text
        x_tick_locations = ax.get_xticks()
        ax.text(x=x_tick_locations[0], y=max_rand_value + (0.0008 * max_rand_value), weight='bold', fontsize=17,
                 s=str(np.round(max_rand_value, 3)))

        #make ths spines thicker:
        #     ax.spines[axis].set_linewidth(1.5)

    if share_y:
        plt.ylim(bottom=data_min - 0.01)
        yticks = np.arange(float(data_min), float(data_max+0.01), 0.02)  # Example: ticks from 0 to 8 with a step of 0.5
        plt.yticks(np.round(yticks, 2))

    # Make the tick lines bigger for all ticks
    plt.sca(axes[0])
    plt.yticks(fontsize=tick_size)
    plt.sca(axes[1])
    plt.yticks(fontsize=tick_size)
    plt.sca(axes[2])
    plt.yticks(fontsize=tick_size)

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.savefig(os.path.join(save_folder, f'combined_sharey{share_y}_no_legend.png'), dpi=300, bbox_inches='tight')

    plt.subplots_adjust(wspace=3.5)
    # Place a single legend next to all subplots
    pos = ax.get_position()
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.94, 0.5), fontsize=14)

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.show()

    export_legend(legend, os.path.join(save_folder, 'combined_legend.png'))

    plt.savefig(os.path.join(save_folder, f'combined_sharey{share_y}.png'), dpi=300, bbox_inches='tight')

    plt.clf()

def plot_graph_for_downstream_scores(save_folder,
                                     df_baseline, enc_df_dict,
                                     x_col_name, y_col_name,
                                     y_label,
                                     title, file_name, skip = 1, max_line=True,
                                     set_upper_lim = True, set_lower_lim = True,
                                     show_baseline = True,
                                     colours_dict = None, linestyle_dict=None,
                                     add_zoom = False, add_x_axis_breaks = False):

    baseline_scores = df_baseline[y_col_name].to_numpy()

    base_rows = df_baseline.shape[0]
    method_rows = list(enc_df_dict.values())[0].shape[0] #- skip

    max_min_base = np.max(baseline_scores)
    if not max_line:
        max_min_base = np.min(baseline_scores)

    #find upper and lower limitsw
    upper_lim = -9999999
    lower_lim = 9999999
    avg_min = 0
    min_list = []
    for key, value in enc_df_dict.items():
        df = value
        df = df[skip:df.shape[0]]
        score_max = df[y_col_name].max()
        score_min = df[y_col_name].min()
        avg_min+=score_min
        if score_min < lower_lim:
            lower_lim = score_min
        if score_max > upper_lim:
            upper_lim = score_max
        min_list.append(score_min)
    avg_min/=len(enc_df_dict)
    #take 1% extra
    upper_lim = upper_lim + (0.01*upper_lim)
    lower_lim = lower_lim - (0.01*lower_lim)

    #recalc lower limit to be the second highest so we can include the first percentage
    if len(min_list) > 2:
        min_list.sort(reverse=True)
        lower_lim = min_list[1] - (0.01*min_list[1])

    plt.clf()
    fig, ax = plt.subplots(figsize=(18, 14), dpi=300)
    if show_baseline:
        plt.axhline(y=max_min_base, color='slategray', label=None, linewidth=7)
        if base_rows >= method_rows:
            plt.plot(df_baseline[x_col_name], df_baseline[y_col_name], marker='*', color='slategray', label=f'Rand', linestyle='solid',
                     linewidth=8, markersize=26)

    for key, value in enc_df_dict.items():
        colour = None
        if colours_dict is not None:
            colour = colours_dict[key]
        linestyle = 'dashed'
        if linestyle_dict is not None:
            linestyle = linestyle_dict[key]
        df = value
        df = df[skip:df.shape[0]]
        plt.plot(df[x_col_name], df[y_col_name], marker='o', color=colour, label=f'{key}', linestyle=linestyle, mew=2,
                 linewidth=8, markersize=18)

    if show_baseline and base_rows < method_rows:
        plt.plot(df_baseline[x_col_name], df_baseline[y_col_name], marker='*', color='slategray', label=f'Rand', linestyle='solid',
                 linewidth=8, markersize=26)

    if set_upper_lim:
        plt.ylim(top=upper_lim)
    if set_lower_lim:
        plt.ylim(bottom=lower_lim)

    label_size = 56#40
    tick_size = 52#36
    inset_tick_size = 34#34
    legend_font_size = 44#30
    plt.xlabel(' Downstream Training Percentage', fontsize=label_size, labelpad=20)
    plt.ylabel(y_label, fontsize=label_size)
    plt.xticks(fontsize=tick_size, ticks=df[x_col_name], labels=df[x_col_name])#, labels=df['percentage_train'])#, rotation=45) #ticks=df['x_label'], labels=df['x_label'])
    plt.yticks(fontsize=tick_size)

    if show_baseline:
        plt.text(x=0, y=max_min_base+(0.0008*max_min_base), weight='bold', fontsize=tick_size,
                 s=str(np.round(max_min_base, 3)))


    # Make the tick lines bigger for all ticks
    plt.rc(('xtick.major', 'ytick.major'), width=10, size=20)

    #make ths spines thicker:
    for axis in ['bottom', 'left']:#['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(4)

    #ax.xaxis.remove_overlapping_locs
    #ax.yaxis.remove_overlapping_locs

    #Add axis breaks to indicate discontinuity
    if add_x_axis_breaks:
        x_min, x_max = ax.get_xlim()
        ticks = [(tick - x_min) / (x_max - x_min) for tick in ax.get_xticks()]

        x_start = ((ticks[-4] + ticks[-5]) / 2) - 0.0045
        axis_break(ax, xpos=[x_start, x_start + 0.015], slant=1.5)



    if add_zoom:
        #data corrdinate to your display coordinate
        #inverted -  from display to data coordinates
        #After the data coordinate system, axes is probably the second most useful coordinate system.
        axis_to_data = ax.transAxes + ax.transData.inverted()

        left, bottom, width, height = [0.51, 0.1700, 0.38, 0.26]
        ax2 = plt.axes([left, bottom, width, height])
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        pos_ax2 =  ax2.get_position()

        for key, value in enc_df_dict.items():
            colour = None
            if colours_dict is not None:
                colour = colours_dict[key]
            linestyle = 'dashed'
            if linestyle_dict is not None:
                linestyle = linestyle_dict[key]

            df1 = value.tail(3)
            ax2.plot(df1[x_col_name], df1[y_col_name], marker='o', color=colour, label=f'{key}', linestyle=linestyle, mew=2,
                     linewidth=4, markersize=12)

        ax2.set_xticks(fontsize=inset_tick_size, ticks=df1[x_col_name], labels=df1[x_col_name])
        ax2.yaxis.set_major_locator(plt.MaxNLocator(2))     #to reduce number of ticks
        for tick in ax2.yaxis.get_major_ticks():
            tick.label.set_fontsize(inset_tick_size)

        plt.sca(ax)
        end_coord = axis_to_data.transform([(pos_ax2.x1, pos_ax2.y1)])
        end_y = end_coord[0][1]
        l1 = [(6, 0.95), (4, end_y)]  # [(6, 0.945), (4, 0.916)]
        l2 = [(8.1, 0.95), (8.24, end_y)]  # [(8.1, 0.945), (8.24, 0.916)]
        lc = LineCollection([l1, l2], color=["gray", "gray"], linestyle='--', lw=1)
        plt.gca().add_collection(lc)

    # Set current axis
    # Because plt.axes adds an Axes to the current figure and makes it the current Axes.
    # To set the current axes, where ax is the Axes object you'd like to become active:
    plt.sca(ax)

    # PLOT LEGEND
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.90])
    legend = plt.legend(fontsize=legend_font_size, loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=3)

    export_legend(legend, os.path.join(save_folder, 'legend.png'))

    plt.savefig(os.path.join(save_folder, file_name), dpi=300, bbox_inches='tight')

    #Hide legend and save figure without legend:
    plt.legend('', frameon=False)
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height])
    file_name_no_legend = file_name[:file_name.rindex('.png')]+'_no_legend.png'
    plt.savefig(os.path.join(save_folder, file_name_no_legend), dpi=300, bbox_inches='tight')

    plt.clf()


def plot_bar_graph_for_downstream_scores(save_folder,
                                     df_baseline, enc_df_dict,
                                     x_col_name, y_col_name,
                                     y_label,
                                     title, file_name, skip = 1, max_line=True,
                                     set_upper_lim = True, set_lower_lim = True,
                                     show_baseline = True,
                                     colours_dict = None, linestyle_dict=None, width=0.20):

    baseline_scores = df_baseline[y_col_name].to_numpy()

    base_rows = df_baseline.shape[0]
    method_rows = list(enc_df_dict.values())[0].shape[0] #- skip

    max_min_base = np.max(baseline_scores)
    if not max_line:
        max_min_base = np.min(baseline_scores)

    #find upper and lower limits
    upper_lim = -9999999
    lower_lim = 9999999
    avg_min = 0
    for key, value in enc_df_dict.items():
        df = value
        df = df[skip:df.shape[0]]
        score_max = df[y_col_name].max()
        score_min = df[y_col_name].min()
        avg_min+=score_min
        if score_min < lower_lim:
            lower_lim = score_min
        if score_max > upper_lim:
            upper_lim = score_max
    avg_min/=len(enc_df_dict)
    #take 1% extra
    upper_lim = upper_lim + (0.01*upper_lim)
    lower_lim = lower_lim - (0.01*lower_lim)

    min_base_lim = np.min(baseline_scores)
    if min_base_lim < lower_lim:
        lower_lim = min_base_lim - (0.01*min_base_lim)

    plt.clf()
    fig, ax = plt.subplots(figsize=(18, 14), dpi=300, layout='constrained')

    if show_baseline:
        plt.axhline(y=max_min_base, color='slategray', label=None, linewidth=4, linestyle=(0, (8, 4)))

    x_labels = df_baseline[x_col_name].tolist()
    data_dict = {}
    data_dict['Rand'] = df_baseline[y_col_name].to_numpy()
    for key, value in enc_df_dict.items():
        df = value
        df = df[skip:df.shape[0]]
        data_dict[key] = df[y_col_name].to_numpy()
    cols_dict = colours_dict.copy()
    cols_dict['Rand'] = 'lightslategray'

    num_bars = len(data_dict)
    x = np.arange(len(x_labels))  # the label locations
    multiplier = 0

    for name, measurement in data_dict.items():
        colour = None
        if cols_dict is not None:
            colour = cols_dict[name]

        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=name, color=colour)
        multiplier += 1

    # set x-ticks
    ax.set_xticks(x + (width*(num_bars/2.2)), x_labels)
    # set the limit for x axis
    x_lims = ax.get_xlim()
    plt.xlim(x_lims[0]+0.3, x_lims[1]-0.3)
    x_lims = ax.get_xlim()

    if set_upper_lim:
        plt.ylim(top=upper_lim)
    if set_lower_lim:
        plt.ylim(bottom=lower_lim)

    ax.yaxis.set_major_locator(plt.MaxNLocator('auto'))  # number of ticks

    label_size = 40#36 #30
    tick_size = 36#32 #26
    inset_tick_size = 34#28 #22
    legend_font_size = 30 #26
    plt.xlabel(' Downstream Training Percentage', fontsize=label_size, labelpad=20)
    plt.ylabel(y_label, fontsize=label_size)
    plt.xticks(fontsize=tick_size)#, labels=df['percentage_train'])#, rotation=45) #ticks=df['x_label'], labels=df['x_label'])
    plt.yticks(fontsize=tick_size)

    # if show_baseline:


    # Make the tick lines bigger for all ticks
    plt.rc(('xtick.major', 'ytick.major'), width=5, size=20)

    #make ths spines thicker:
    for axis in ['bottom', 'left']:#['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(4)

    # PLOT LEGEND
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.90])
    legend = plt.legend(fontsize=legend_font_size, loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=3)

    export_legend(legend, os.path.join(save_folder, 'legend.png'))

    plt.savefig(os.path.join(save_folder, file_name), dpi=300, bbox_inches='tight')

    #Hide legend and save figure without legend:
    plt.legend('', frameon=False)
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height])
    file_name_no_legend = file_name[:file_name.rindex('.png')]+'_no_legend.png'
    plt.savefig(os.path.join(save_folder, file_name_no_legend), dpi=300, bbox_inches='tight')

    plt.clf()

def write_results_for_methods_comparison_new_plots(save_folder, main_folders_list, names, encoder_percentage,
                                         show_baseline = True,
                                         colours_dict = None,
                                         linestyle_dict = None,
                                         baseline_folder = 'Baseline_444_Final'):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    df_base = get_baseline_scores_df('.', baseline_folder)

    percentages_list = [1, 2, 3, 4, 5, 10, 15, 25, 100]
    mask = df_base['percentage_train'].isin(percentages_list)
    df_base = df_base[mask]

    dfs_dict = {}

    dtype_dict = {
            'percentage_unlabelled': 'int64',
            'num_unlabelled': 'int64',
            'percentage_train': 'float64',
            'train': 'int64',
            'dice_test': 'float64',
            'dice_test_std': 'float64',
            'dice_con': 'float64',
            'dice_con_std': 'float64',
            'dice_con_top100': 'float64',
            'dice_con_std_top100': 'float64',
            'hd_test': 'float64',
            'hd_test_std': 'float64',
            'hd_con': 'float64',
            'hd_con_std': 'float64',
            'hd_con_top100': 'float64',
            'hd_con_std_top100': 'float64',
            'val': 'int64',
            'test': 'int64',
            'consensus': 'int64',
            'x_label': 'str',
            'hue_label': 'str',
            'ef_error': 'float64',
            'ef_error_std': 'float64',
            'ed_vol_error': 'float64',
            'ed_vol_error_std': 'float64',
            'es_vol_error': 'float64',
            'es_vol_error_std': 'float64'
    }

    for i in range(len(main_folders_list)):
        main_folder = main_folders_list[i]
        name = names[i]
        if not os.path.exists(main_folder):
            continue

        files = os.listdir(main_folder)
        for file in files:
            if not 'results_on_enc' in file: continue

            df = pd.read_csv(os.path.join(main_folder, file), dtype=dtype_dict)
            mask = df['percentage_train'].isin(percentages_list)
            df = df[mask]

            unlabelled_percentage = df.iloc[0]['percentage_unlabelled']
            #     continue

            dfs_dict[name] = df
            break

    if len(dfs_dict) == 0:
        return

    num = df.iloc[0]['num_unlabelled']

    #reorder:
    #When plotting more than one graph in matplotlib, the graph with the most x-coordinates
    #should be plotted first, since we're using string x-labels, matplotlib cannot re-order
    #it automatically, so you may get 10% x-value coming after 100%.
    if len(dfs_dict) > 1:
        dfs_dict_reorder = {}
        max_rows = 0
        name = None
        for key, value in dfs_dict.items():
            if value.shape[0]> max_rows:
                max_rows = value.shape[0]
                name = key
        dfs_dict_reorder[name] = dfs_dict[name]
        for key, value in dfs_dict.items():
            if key not in dfs_dict_reorder:
                dfs_dict_reorder[key] = value
        dfs_dict = dfs_dict_reorder

    skip_num = 0
    skip_base_num = 0

    plot_graph_for_downstream_scores_combined(save_folder,
                                                  df_baseline=df_base[skip_base_num:], enc_df_dict=dfs_dict,
                                                  colours_dict=colours_dict,
                                                  linestyle_dict=linestyle_dict
                                                  )
    return

def write_results_for_methods_comparison_new_plots_test_only(save_folder, main_folders_list, names, encoder_percentage,
                                         show_baseline = True,
                                         colours_dict = None,
                                         linestyle_dict = None,
                                         baseline_folder = 'Baseline_444_Final'):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    df_base = get_baseline_scores_df('.', baseline_folder)

    percentages_list = [1, 2, 3, 4, 5, 10, 15, 25, 100]
    mask = df_base['percentage_train'].isin(percentages_list)
    df_base = df_base[mask]

    dfs_dict = {}

    dtype_dict = {
            'percentage_unlabelled': 'int64',
            'num_unlabelled': 'int64',
            'percentage_train': 'float64',
            'train': 'int64',
            'dice_test': 'float64',
            'dice_test_std': 'float64',
            'dice_con': 'float64',
            'dice_con_std': 'float64',
            'dice_con_top100': 'float64',
            'dice_con_std_top100': 'float64',
            'hd_test': 'float64',
            'hd_test_std': 'float64',
            'hd_con': 'float64',
            'hd_con_std': 'float64',
            'hd_con_top100': 'float64',
            'hd_con_std_top100': 'float64',
            'val': 'int64',
            'test': 'int64',
            'consensus': 'int64',
            'x_label': 'str',
            'hue_label': 'str',
            'ef_error': 'float64',
            'ef_error_std': 'float64',
            'ed_vol_error': 'float64',
            'ed_vol_error_std': 'float64',
            'es_vol_error': 'float64',
            'es_vol_error_std': 'float64'
    }

    for i in range(len(main_folders_list)):
        main_folder = main_folders_list[i]
        name = names[i]
        if not os.path.exists(main_folder):
            continue

        files = os.listdir(main_folder)
        for file in files:
            if not 'results_on_enc' in file: continue

            df = pd.read_csv(os.path.join(main_folder, file), dtype=dtype_dict)
            mask = df['percentage_train'].isin(percentages_list)
            df = df[mask]

            unlabelled_percentage = df.iloc[0]['percentage_unlabelled']
            #     continue

            dfs_dict[name] = df
            break

    if len(dfs_dict) == 0:
        return

    num = df.iloc[0]['num_unlabelled']

    #reorder:
    #When plotting more than one graph in matplotlib, the graph with the most x-coordinates
    #should be plotted first, since we're using string x-labels, matplotlib cannot re-order
    #it automatically, so you may get 10% x-value coming after 100%.
    if len(dfs_dict) > 1:
        dfs_dict_reorder = {}
        max_rows = 0
        name = None
        for key, value in dfs_dict.items():
            if value.shape[0]> max_rows:
                max_rows = value.shape[0]
                name = key
        dfs_dict_reorder[name] = dfs_dict[name]
        for key, value in dfs_dict.items():
            if key not in dfs_dict_reorder:
                dfs_dict_reorder[key] = value
        dfs_dict = dfs_dict_reorder

    skip_num = 0
    skip_base_num = 0

    plot_graph_for_downstream_scores_combined_test_only(save_folder,
                                                  df_baseline=df_base[skip_base_num:], enc_df_dict=dfs_dict,
                                                  colours_dict=colours_dict,
                                                  linestyle_dict=linestyle_dict
                                                  )
    return

def write_results_for_methods_comparison(save_folder, main_folders_list, names, encoder_percentage,
                                         show_baseline = True,
                                         colours_dict = None,
                                         linestyle_dict = None,
                                         bar_width = 0.20):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    df_base = get_baseline_scores_df('.')

    percentages_list = [1, 2, 3, 4, 5, 10, 15, 25, 100]
    mask = df_base['percentage_train'].isin(percentages_list)
    df_base = df_base[mask]

    dfs_dict = {}

    dtype_dict = {
            'percentage_unlabelled': 'int64',
            'num_unlabelled': 'int64',
            'percentage_train': 'float64',
            'train': 'int64',
            'dice_test': 'float64',
            'dice_test_std': 'float64',
            'dice_con': 'float64',
            'dice_con_std': 'float64',
            'dice_con_top100': 'float64',
            'dice_con_std_top100': 'float64',
            'hd_test': 'float64',
            'hd_test_std': 'float64',
            'hd_con': 'float64',
            'hd_con_std': 'float64',
            'hd_con_top100': 'float64',
            'hd_con_std_top100': 'float64',
            'val': 'int64',
            'test': 'int64',
            'consensus': 'int64',
            'x_label': 'str',
            'hue_label': 'str',
            'ef_error': 'float64',
            'ef_error_std': 'float64',
            'ed_vol_error': 'float64',
            'ed_vol_error_std': 'float64',
            'es_vol_error': 'float64',
            'es_vol_error_std': 'float64'
    }

    for i in range(len(main_folders_list)):
        main_folder = main_folders_list[i]
        name = names[i]
        if not os.path.exists(main_folder):
            continue

        files = os.listdir(main_folder)
        for file in files:
            if not 'results_on_enc' in file: continue

            df = pd.read_csv(os.path.join(main_folder, file), dtype=dtype_dict)
            mask = df['percentage_train'].isin(percentages_list)
            df = df[mask]

            unlabelled_percentage = df.iloc[0]['percentage_unlabelled']
            #     continue

            dfs_dict[name] = df
            break

    if len(dfs_dict) == 0:
        return

    num = df.iloc[0]['num_unlabelled']

    #reorder:
    #When plotting more than one graph in matplotlib, the graph with the most x-coordinates
    #should be plotted first, since we're using string x-labels, matplotlib cannot re-order
    #it automatically, so you may get 10% x-value coming after 100%.
    if len(dfs_dict) > 1:
        dfs_dict_reorder = {}
        max_rows = 0
        name = None
        for key, value in dfs_dict.items():
            if value.shape[0]> max_rows:
                max_rows = value.shape[0]
                name = key
        dfs_dict_reorder[name] = dfs_dict[name]
        for key, value in dfs_dict.items():
            if key not in dfs_dict_reorder:
                dfs_dict_reorder[key] = value
        dfs_dict = dfs_dict_reorder

    skip_num = 0
    skip_base_num = 0

    # Write dice graph for test set
    plot_graph_for_downstream_scores(save_folder=save_folder, df_baseline=df_base[skip_base_num:], enc_df_dict=dfs_dict,
                                     x_col_name='x_label', y_col_name='dice_test', y_label='Dice',
                                     title=f'Test Set Results - encoder-{num}',
                                     file_name=f'dice_results_test_{encoder_percentage}.png',
                                     skip=skip_num,
                                     max_line=True,
                                     set_upper_lim=False, set_lower_lim=True,
                                     show_baseline=show_baseline,
                                     colours_dict=colours_dict,
                                     linestyle_dict=linestyle_dict,
                                     add_zoom=False, add_x_axis_breaks=True)
    plot_bar_graph_for_downstream_scores(save_folder=save_folder, df_baseline=df_base[skip_base_num:], enc_df_dict=dfs_dict,
                                     x_col_name='x_label', y_col_name='dice_test', y_label='Dice',
                                     title=f'Test Set Results - encoder-{num}',
                                     file_name=f'dice_results_test_{encoder_percentage}_BAR.png',
                                     skip=skip_num,
                                     max_line=True,
                                     set_upper_lim=True, set_lower_lim=True,
                                     show_baseline=show_baseline,
                                     colours_dict=colours_dict,
                                     linestyle_dict=linestyle_dict, width = bar_width)

    # Write dice graph for consensus set
    plot_graph_for_downstream_scores(save_folder=save_folder, df_baseline=df_base[skip_base_num:], enc_df_dict=dfs_dict,
                                     x_col_name='x_label', y_col_name='dice_con', y_label='Dice',
                                     title=f'Consensus Set Results - encoder-{num}',
                                     file_name=f'dice_results_consensus_{encoder_percentage}.png',
                                     skip=skip_num,
                                     max_line=True,
                                     set_upper_lim=False, set_lower_lim=True,
                                     show_baseline=show_baseline,
                                     colours_dict=colours_dict,
                                     linestyle_dict=linestyle_dict,
                                     add_zoom=False, add_x_axis_breaks=True)
    plot_bar_graph_for_downstream_scores(save_folder=save_folder, df_baseline=df_base[skip_base_num:], enc_df_dict=dfs_dict,
                                     x_col_name='x_label', y_col_name='dice_con', y_label='Dice',
                                     title=f'Consensus Set Results - encoder-{num}',
                                     file_name=f'dice_results_consensus_{encoder_percentage}_BAR.png',
                                     skip=skip_num,
                                     max_line=True,
                                     set_upper_lim=True, set_lower_lim=True,
                                     show_baseline=show_baseline,
                                     colours_dict=colours_dict,
                                     linestyle_dict=linestyle_dict, width = bar_width)

    plot_graph_for_downstream_scores(save_folder=save_folder, df_baseline=df_base[skip_base_num:], enc_df_dict=dfs_dict,
                                     x_col_name='x_label', y_col_name='dice_con_top100', y_label='Dice',
                                     title=f'Consensus Set Results - encoder-{num} - Top 100',
                                     file_name=f'dice_results_consensus_{encoder_percentage}_top100.png',
                                     skip=skip_num,
                                     max_line=True,
                                     set_upper_lim=False, set_lower_lim=True,
                                     show_baseline=show_baseline,
                                     colours_dict=colours_dict,
                                     linestyle_dict=linestyle_dict,
                                     add_zoom=False, add_x_axis_breaks=True)
    plot_bar_graph_for_downstream_scores(save_folder=save_folder, df_baseline=df_base[skip_base_num:], enc_df_dict=dfs_dict,
                                     x_col_name='x_label', y_col_name='dice_con_top100', y_label='Dice',
                                     title=f'Consensus Set Results - encoder-{num} - Top 100',
                                     file_name=f'dice_results_consensus_{encoder_percentage}_top100_BAR.png',
                                     skip=skip_num,
                                     max_line=True,
                                     set_upper_lim=True, set_lower_lim=True,
                                     show_baseline=show_baseline,
                                     colours_dict=colours_dict,
                                     linestyle_dict=linestyle_dict, width = bar_width)
    return
    # Write HD graph for test set
    plot_graph_for_downstream_scores(save_folder=save_folder, df_baseline=df_base[skip_base_num:], enc_df_dict=dfs_dict,
                                     x_col_name='x_label', y_col_name='hd_test', y_label='HD',
                                     title=f'Test Set Results - encoder-{num}',
                                     file_name=f'hd_results_test_{encoder_percentage}.png',
                                     skip=skip_num,
                                     max_line=False,
                                     set_upper_lim=True, set_lower_lim=False,
                                     show_baseline=show_baseline,
                                     colours_dict=colours_dict,
                                     linestyle_dict=linestyle_dict)

    # Write HD graph for consensus set
    plot_graph_for_downstream_scores(save_folder=save_folder, df_baseline=df_base[skip_base_num:], enc_df_dict=dfs_dict,
                                     x_col_name='x_label', y_col_name='hd_con', y_label='HD',
                                     title=f'Consensus Set Results - encoder-{num}',
                                     file_name=f'hd_results_consensus_{encoder_percentage}.png',
                                     skip=skip_num,
                                     max_line=False,
                                     set_upper_lim=True, set_lower_lim=False,
                                     show_baseline=show_baseline,
                                     colours_dict=colours_dict,
                                     linestyle_dict=linestyle_dict)
    plot_graph_for_downstream_scores(save_folder=save_folder, df_baseline=df_base[skip_base_num:], enc_df_dict=dfs_dict,
                                     x_col_name='x_label', y_col_name='hd_con_top100', y_label='HD',
                                     title=f'Consensus Set Results - encoder-{num} - Top 100',
                                     file_name=f'hd_results_consensus_{encoder_percentage}_top100.png',
                                     skip=skip_num,
                                     max_line=False,
                                     set_upper_lim=True, set_lower_lim=False,
                                     show_baseline=show_baseline,
                                     colours_dict=colours_dict,
                                     linestyle_dict=linestyle_dict)

    # EJECTION FRACTION
    # Write Ejection Fraction graph for consensus set
    plot_graph_for_downstream_scores(save_folder=save_folder, df_baseline=df_base[skip_base_num:],
                                     enc_df_dict=dfs_dict,
                                     x_col_name='x_label', y_col_name='ef_error', y_label='EF error',
                                     title=f'Consensus Set Results - encoder-{num}',
                                     file_name=f'EF_error_results_consensus_{encoder_percentage}.png',
                                     skip=skip_num,
                                     max_line=False,
                                     set_upper_lim=True, set_lower_lim=False,
                                     show_baseline=show_baseline,
                                     colours_dict=colours_dict,
                                     linestyle_dict=linestyle_dict)

    # Write ED volume graph for consensus set
    plot_graph_for_downstream_scores(save_folder=save_folder, df_baseline=df_base[skip_base_num:],
                                     enc_df_dict=dfs_dict,
                                     x_col_name='x_label', y_col_name='ed_vol_error', y_label='EDV error',
                                     title=f'Consensus Set Results - encoder-{num}',
                                     file_name=f'EDV_error_results_consensus_{encoder_percentage}.png',
                                     skip=skip_num,
                                     max_line=False,
                                     set_upper_lim=True, set_lower_lim=False,
                                     show_baseline=show_baseline,
                                     colours_dict=colours_dict,
                                     linestyle_dict=linestyle_dict)

    # Write ES volume graph for consensus set
    plot_graph_for_downstream_scores(save_folder=save_folder, df_baseline=df_base[skip_base_num:],
                                     enc_df_dict=dfs_dict,
                                     x_col_name='x_label', y_col_name='es_vol_error', y_label='ESV error',
                                     title=f'Consensus Set Results - encoder-{num}',
                                     file_name=f'ESV_error_results_consensus_{encoder_percentage}.png',
                                     skip=skip_num,
                                     max_line=False,
                                     set_upper_lim=True, set_lower_lim=False,
                                     show_baseline=show_baseline,
                                     colours_dict=colours_dict,
                                     linestyle_dict=linestyle_dict)


def write_avg_results_per_encoder_percentage(main_folder, title, baseline_folder='Baseline_444_Final'):
    plt.clf()

    exp_folders = os.listdir(main_folder)
    exp_folders.sort()

    # Get the average baseline.
    df_base = get_baseline_scores_df(os.path.relpath('.'), baseline_folder)

    # save df as csv
    df_base.sort_values(by=['percentage_train'], inplace=True)
    df_base.to_csv(os.path.join(main_folder, f'baseline_results.csv'))

    # Get results for each encoder at different percentages of downstream training data
    results_dict = {}
    for i in range(len(exp_folders)):
        if 'exp_with' not in exp_folders[i].lower():
            continue

        exp_folder = os.path.join(main_folder, exp_folders[i])
        downstream_folders = os.listdir(exp_folder)

        for downstream_folder in downstream_folders:
            if 'finetuning_' not in downstream_folder.lower():
                continue

            results_folder = os.path.join(exp_folder, downstream_folder)
            if not os.path.exists(results_folder):
                continue

            files = os.listdir(results_folder)

            res_file = 'encoder_fine_tune_results.json'
            res_ef_file = 'Consensus_EF_Vol_error_results.json'
            if not (res_file in files or res_ef_file in files):
                continue

            res_dict = load_json(os.path.join(results_folder, res_file))

            percentage_unlabelled = res_dict['percentage_unlabelled']
            percentage_train = res_dict['percentage_train']
            file_with_path = os.path.join(results_folder, res_file)

            if percentage_unlabelled in results_dict:
                results = results_dict[percentage_unlabelled]
                results[0].append(percentage_train)
                results[1].append(file_with_path)
                results[2].append(os.path.join(results_folder, res_ef_file))
            else:
                percentages, res_files, res_ef_files = [percentage_train], [file_with_path], [os.path.join(results_folder, res_ef_file)]
                results_dict[percentage_unlabelled] = [percentages, res_files, res_ef_files]

    results_dict = dict(sorted(results_dict.items()))
    enc_df_dict = {}
    for key, value in results_dict.items():
        percs = value[0]
        res_files = value[1]
        res_ef_files = value[2]
        percentages, results_files, results_ef_files = zip(*sorted(zip(percs, res_files, res_ef_files)))

        df = pd.DataFrame(columns=['percentage_unlabelled', 'num_unlabelled',
                                   'percentage_train', 'train',
                                   'dice_test', 'dice_test_std',
                                   'dice_con', 'dice_con_std',
                                   'dice_con_top100', 'dice_con_std_top100',
                                   'hd_test', 'hd_test_std',
                                   'hd_con', 'hd_con_std',
                                   'hd_con_top100', 'hd_con_std_top100',
                                   'iou_test', 'iou_test_std',
                                   'iou_con', 'iou_con_std',
                                   'iou_con_top100', 'iou_con_std_top100',
                                   'val', 'test', 'consensus',
                                   'x_label', 'hue_label',
                                   'ef_error', 'ef_error_std',
                                   'ed_vol_error', 'ed_vol_error_std',
                                   'es_vol_error', 'es_vol_error_std'])

        df['percentage_unlabelled'] = df['percentage_unlabelled'].astype(int)
        df['num_unlabelled'] = df['num_unlabelled'].astype(int)
        df['percentage_train'] = df['percentage_train'].astype(int)
        df['train'] = df['train'].astype(int)

        df['dice_test'] = df['dice_test'].astype(float)
        df['dice_test_std'] = df['dice_test_std'].astype(float)
        df['dice_con'] = df['dice_con'].astype(float)
        df['dice_con_std'] = df['dice_con_std'].astype(float)
        df['dice_con_top100'] = df['dice_con_top100'].astype(float)
        df['dice_con_std_top100'] = df['dice_con_std_top100'].astype(float)

        df['hd_test'] = df['hd_test'].astype(float)
        df['hd_test_std'] = df['hd_test_std'].astype(float)
        df['hd_con'] = df['hd_con'].astype(float)
        df['hd_con_std'] = df['hd_con_std'].astype(float)
        df['hd_con_top100'] = df['hd_con_top100'].astype(float)
        df['hd_con_std_top100'] = df['hd_con_std_top100'].astype(float)

        df['iou_test'] = df['iou_test'].astype(float)
        df['iou_test_std'] = df['iou_test_std'].astype(float)
        df['iou_con'] = df['iou_con'].astype(float)
        df['iou_con_std'] = df['iou_con_std'].astype(float)
        df['iou_con_top100'] = df['iou_con_top100'].astype(float)
        df['iou_con_std_top100'] = df['iou_con_std_top100'].astype(float)

        df['val'] = df['val'].astype(int)
        df['test'] = df['test'].astype(int)
        df['consensus'] = df['consensus'].astype(int)
        df['x_label'] = df['x_label'].astype(str)
        df['hue_label'] = df['hue_label'].astype(str)
        df['ef_error'] = df['ef_error'].astype(float)
        df['ef_error_std'] = df['ef_error_std'].astype(float)
        df['ed_vol_error'] = df['ed_vol_error'].astype(float)
        df['ed_vol_error_std'] = df['ed_vol_error_std'].astype(float)
        df['es_vol_error'] = df['es_vol_error'].astype(float)
        df['es_vol_error_std'] = df['es_vol_error_std'].astype(float)

        # append row to df
        num_unlabelled = None
        percentage_unlabelled = None
        # accross different amounts of downstream labelled training data.
        for i, file in enumerate(results_files):
            res_dict = load_json(file)

            run_dices_test = np.array(res_dict['run_dices_test'])
            run_hds_test = np.array(res_dict['run_hds_test'])
            run_iou_test = np.array(res_dict['run_iou_test'])

            run_dices_consensus = np.array(res_dict['run_dices_consensus'])
            run_hds_consensus = np.array(res_dict['run_hds_consensus'])
            run_iou_consensus = np.array(res_dict['run_iou_consensus'])

            run_dices_consensus_top100 = np.array(res_dict['run_dices_consensus_top100'])
            run_hds_consensus_top100 = np.array(res_dict['run_hds_consensus_top100'])
            run_iou_consensus_top100 = np.array(res_dict['run_iou_consensus_top100'])

            num_unlabelled = res_dict['num_unlabelled']
            percentage_unlabelled = res_dict['percentage_unlabelled']
            num_train = res_dict['num_train']
            percentage_train = res_dict['percentage_train']
            num_val = res_dict['num_val']
            num_test = res_dict['num_test']
            num_consensus = res_dict['num_consensus']
            x_label = str(percentage_train) #f'{num_train}({percentage_train}%)'
            hue_label = f'enc_{num_unlabelled}'

            res_ef_dict = load_json(results_ef_files[i])
            ef_errors = np.array(res_ef_dict['avg_ef_errors_per_run'])
            ed_errors = np.array(res_ef_dict['avg_ed_vol_errors_per_run'])
            es_errors = np.array(res_ef_dict['avg_es_vol_errors_per_run'])

            row = {'percentage_unlabelled': percentage_unlabelled, 'num_unlabelled': num_unlabelled,
                   'dice_test': np.mean(run_dices_test), 'dice_test_std': np.std(run_dices_test),
                   'dice_con': np.mean(run_dices_consensus), 'dice_con_std': np.std(run_dices_consensus),
                   'dice_con_top100': np.mean(run_dices_consensus_top100), 'dice_con_std_top100': np.std(run_dices_consensus_top100),
                   'hd_test': np.mean(run_hds_test), 'hd_test_std': np.std(run_hds_test),
                   'hd_con': np.mean(run_hds_consensus), 'hd_con_std': np.std(run_hds_consensus),
                   'hd_con_top100': np.mean(run_hds_consensus_top100), 'hd_con_std_top100': np.std(run_hds_consensus_top100),
                   'iou_test': np.mean(run_iou_test), 'iou_test_std': np.std(run_iou_test),
                   'iou_con': np.mean(run_iou_consensus), 'iou_con_std': np.std(run_iou_consensus),
                   'iou_con_top100': np.mean(run_iou_consensus_top100), 'iou_con_std_top100': np.std(run_iou_consensus_top100),
                   'percentage_train': percentage_train, 'train': num_train, 'val': num_val,
                   'test': num_test, 'consensus': num_consensus, 'best_file': file,
                   'x_label': x_label, 'hue_label': hue_label,
                   'ef_error': np.mean(ef_errors), 'ef_error_std': np.std(ef_errors),
                   'ed_vol_error': np.mean(ed_errors), 'ed_vol_error_std': np.std(ed_errors),
                   'es_vol_error': np.mean(es_errors), 'es_vol_error_std': np.std(es_errors)
                   }
            df = df.append(row, ignore_index=True)

        # save df as csv
        df.to_csv(os.path.join(main_folder, f'results_on_enc_{percentage_unlabelled}.csv'))

        enc_df_dict[f'enc-{percentage_unlabelled}'] = df

    if len(enc_df_dict) == 0:
        return

    skip_num = 1
    skip_base_num = 1
    # Write dice graph for test set
    plot_graph_for_downstream_scores(save_folder=main_folder, df_baseline=df_base[skip_base_num:], enc_df_dict=enc_df_dict,
                                     x_col_name='x_label', y_col_name='dice_test', y_label='Dice',
                                     title=f'Test Set Results - {title}', file_name='dice_results_test.png',
                                     skip=skip_num,
                                     max_line=True,
                                     set_upper_lim=False, set_lower_lim=True)

    # Write dice graph for consensus set
    plot_graph_for_downstream_scores(save_folder=main_folder, df_baseline=df_base[skip_base_num:],
                                     enc_df_dict=enc_df_dict, x_col_name='x_label', y_col_name='dice_con',
                                     y_label='Dice', title=f'Consensus Set Results - {title}',
                                     file_name='dice_results_consensus.png',
                                     skip=skip_num,
                                     max_line=True,
                                     set_upper_lim=False, set_lower_lim=True)
    plot_graph_for_downstream_scores(save_folder=main_folder, df_baseline=df_base[skip_base_num:],
                                     enc_df_dict=enc_df_dict, x_col_name='x_label', y_col_name='dice_con_top100',
                                     y_label='Dice', title=f'Consensus Set Results - {title} - Top 100',
                                     file_name='dice_results_consensus_top100.png',
                                     skip=skip_num,
                                     max_line=True,
                                     set_upper_lim=False, set_lower_lim=True)

    # Write HD graph for test set
    plot_graph_for_downstream_scores(save_folder=main_folder, df_baseline=df_base[skip_base_num:],
                                     enc_df_dict=enc_df_dict, x_col_name='x_label', y_col_name='hd_test', y_label='HD',
                                     title=f'Test Set Results - {title}', file_name='hd_results_test.png',
                                     skip=skip_num,
                                     max_line=False,
                                     set_upper_lim=True, set_lower_lim=False)

    # Write HD graph for consensus set
    plot_graph_for_downstream_scores(save_folder=main_folder, df_baseline=df_base[skip_base_num:],
                                     enc_df_dict=enc_df_dict, x_col_name='x_label', y_col_name='hd_con', y_label='HD',
                                     title=f'Consensus Set Results - {title}', file_name='hd_results_consensus.png',
                                     skip=skip_num,
                                     max_line=False,
                                     set_upper_lim=True, set_lower_lim=False)
    plot_graph_for_downstream_scores(save_folder=main_folder, df_baseline=df_base[skip_base_num:],
                                     enc_df_dict=enc_df_dict, x_col_name='x_label', y_col_name='hd_con_top100',
                                     y_label='HD', title=f'Consensus Set Results - {title} - Top 100',
                                     file_name='hd_results_consensus_top100.png',
                                     skip=skip_num,
                                     max_line=False,
                                     set_upper_lim=True, set_lower_lim=False)

    return
    #EJECTION
    skip_num = 1
    skip_base_num = 1
    # Write Ejection Fraction graph for consensus set
    plot_graph_for_downstream_scores(save_folder=main_folder, df_baseline=df_base[skip_base_num:], enc_df_dict=enc_df_dict,
                                     x_col_name='x_label', y_col_name='ef_error', y_label='EF error',
                                     title=f'Consensus Set Results - {title}', file_name='EF_err_results_consensus.png',
                                     skip=skip_num,
                                     max_line=False,
                                     set_upper_lim=False, set_lower_lim=False)

    # Write ED volume graph for consensus set
    plot_graph_for_downstream_scores(save_folder=main_folder, df_baseline=df_base[skip_base_num:], enc_df_dict=enc_df_dict,
                                     x_col_name='x_label', y_col_name='ed_vol_error', y_label='EDV error',
                                     title=f'Consensus Set Results - {title}',
                                     file_name='ED_vol_err_results_consensus.png',
                                     skip=skip_num,
                                     max_line=False,
                                     set_upper_lim=False, set_lower_lim=False)

    # Write ES volume graph for consensus set
    plot_graph_for_downstream_scores(save_folder=main_folder, df_baseline=df_base[skip_base_num:], enc_df_dict=enc_df_dict,
                                     x_col_name='x_label', y_col_name='es_vol_error', y_label='ESV error',
                                     title=f'Consensus Set Results - {title}',
                                     file_name='ES_vol_err_results_consensus.png',
                                     skip=skip_num,
                                     max_line=False,
                                     set_upper_lim=False, set_lower_lim=False)

    ##############################################################################
    ##############################################################################
    # PLOT A BAR GRAPH
    df_all = pd.concat(enc_df_dict.values())
    df_all = pd.concat([df_base, df_all])

    ax = sns.catplot(height=12, aspect=1.8,
                     x="x_label",  # x variable name
                     y="dice_con",  # y variable name
                     hue="hue_label",  # group variable name
                     data=df_all,  # dataframe to plot
                     kind="bar",
                     )
    plt.tight_layout()
    plt.setp(ax._legend.get_texts(), fontsize='20')
    plt.title(f'Consensus Set Results - {title}', fontsize=22)
    plt.xlabel(' No. Labelled Training Data', fontsize=22)
    plt.ylabel('Dice', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(os.path.join(main_folder, 'dice_results_consensus_bar.png'), bbox_inches='tight')
    plt.clf()

    ax = sns.catplot(height=12, aspect=1.8,
                     x="x_label",  # x variable name
                     y="dice_test",  # y variable name
                     hue="hue_label",  # group variable name
                     data=df_all,  # dataframe to plot
                     kind="bar",
                     )
    plt.setp(ax._legend.get_texts(), fontsize='20')
    plt.title(f'Test Set Results - {title}', fontsize=22)
    plt.xlabel(' No. Labelled Training Data', fontsize=22)
    plt.ylabel('Dice', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(os.path.join(main_folder, 'dice_results_test_bar.png'), bbox_inches='tight')
    plt.clf()

    ax = sns.catplot(height=12, aspect=1.8,
                     x="x_label",  # x variable name
                     y="hd_con",  # y variable name
                     hue="hue_label",  # group variable name
                     data=df_all,  # dataframe to plot
                     kind="bar",
                     )
    plt.setp(ax._legend.get_texts(), fontsize='20')
    plt.title(f'Consensus Set Results - {title}', fontsize=22)
    plt.xlabel(' No. Labelled Training Data', fontsize=22)
    plt.ylabel('HD', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(os.path.join(main_folder, 'hd_results_consensus_bar.png'), bbox_inches='tight')
    plt.clf()

    ax = sns.catplot(height=12, aspect=1.8,
                     x="x_label",  # x variable name
                     y="hd_test",  # y variable name
                     hue="hue_label",  # group variable name
                     data=df_all,  # dataframe to plot
                     kind="bar",
                     )
    plt.setp(ax._legend.get_texts(), fontsize='20')
    plt.title(f'Test Set Results - {title}', fontsize=22)
    plt.xlabel(' No. Labelled Training Data', fontsize=22)
    plt.ylabel('HD', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(os.path.join(main_folder, 'hd_results_test_bar.png'), bbox_inches='tight')
    plt.clf()



def write_results_for_methods_comparison_distributions(save_folder, main_folders_list, names, encoder_percentage, show_baseline = True, colours_dict = None, linestyle_dict = None):
    plt.clf()

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    dfs_dict = {}

    for i in range(len(main_folders_list)):
        main_folder = main_folders_list[i]
        name = names[i]
        if not os.path.exists(main_folder):
            continue

        exp_folders = os.listdir(main_folder)
        exp_folders.sort()

        for j in range(len(exp_folders)):
            if 'exp_with' not in exp_folders[j].lower():
                continue

            e_folder = exp_folders[j]

            unlabelled_count = int(e_folder[9:e_folder.index('_percent')])
            if unlabelled_count != encoder_percentage:
                continue

            exp_folder = os.path.join(main_folder, exp_folders[j])
            downstream_folders = os.listdir(exp_folder)


            percentage_unlabelled = None

            for downstream_folder in downstream_folders:
                if 'finetuning_' not in downstream_folder.lower():
                    continue

                results_folder = os.path.join(exp_folder, downstream_folder)
                if not os.path.exists(results_folder):
                    continue

                files = os.listdir(results_folder)

                res_file = 'encoder_fine_tune_results.json'
                res_ef_file = f'Consensus_EF_Vol_results_all_{0}.json'
                res_ef_file_run_1 = f'Consensus_EF_Vol_results_all_{1}.json'
                if not (res_file in files or res_ef_file in files):
                    continue

                res_dict = load_json(os.path.join(results_folder, res_file))

                percentage_unlabelled = res_dict['percentage_unlabelled']
                percentage_train = res_dict['percentage_train']
                num_train = res_dict['num_train']

                dict1 = load_json(os.path.join(results_folder, res_ef_file))
                dict2 = load_json(os.path.join(results_folder, res_ef_file_run_1))
                ef_err_list, edv_err_list, esv_err_list = [], [], []
                ef_list, edv_list, esv_list = [], [], []
                for key, value in dict1.items():
                    value1 = dict2[key]

                    if value['Ejection Fraction pred'] == 0 or value1['Ejection Fraction pred'] == 0:
                        continue

                    ef_err_list.append((value['EF error'] + value1['EF error']) / 2.0)
                    edv_err_list.append((value['EDV error'] + value1['EDV error']) / 2.0)
                    esv_err_list.append((value['ESV error'] + value1['ESV error']) / 2.0)
                    ef_list.append((value['Ejection Fraction pred'] + value1['Ejection Fraction pred']) / 2.0)
                    edv_list.append((value['EDV pred'] + value1['EDV pred']) / 2.0)
                    esv_list.append((value['ESV pred'] + value1['ESV pred']) / 2.0)

                x_tick = f'{num_train}({percentage_train}%)'

                percentage_dict = {}
                if percentage_train in dfs_dict:
                    percentage_dict =  dfs_dict[percentage_train]
                else:
                    dfs_dict[percentage_train] = percentage_dict

                percentage_dict[names[i]] = (x_tick, ef_err_list, edv_err_list, esv_err_list, ef_list, edv_list, esv_list)

    if len(dfs_dict) == 0:
        return

    for key,value in dfs_dict.items():

        percentage_train = key

        data_lists = []
        names = []
        colours = []
        for key2, method_results in value.items():
            method_name = key2
            colours.append(colours_dict[method_name])
            ef_error_list = method_results[1]
            data_lists.append(ef_error_list)
            names.append(method_name)

        fig, ax = plt.subplots()
        bplot = ax.boxplot(data_lists, patch_artist=True, labels=names, showmeans=True, meanline=True, vert=False)

        # 'boxes', 'whiskers', 'fliers', 'medians', 'caps'

        if colours_dict is not None:
            for patch, colour in zip(bplot['boxes'], colours):
                patch.set(facecolor=colour)

            for flier, colour in zip(bplot['fliers'], colours):
                flier.set(markeredgecolor=colour)


        # if base_lines_pos is not None:
        #
        #             continue
        #

        ax.xaxis.grid(True)

        plt.xlabel('EF Error')
        plt.ylabel('Method')

        plt.savefig(os.path.join(save_folder, f'box_plot_{percentage_train}.png'), dpi=300)
        plt.clf()

def create_box_plot(save_path, data_lists, colours_list, x_label, y_label, x_ticks_list, base_lines = None, base_lines_pos = None):

    fig, ax = plt.subplots()
    bplot = ax.boxplot(data_lists, patch_artist=False, labels=x_ticks_list, showmeans=True, meanline=True)

    #'boxes', 'whiskers', 'fliers', 'medians', 'caps'

    if colours_list is not None:
        for patch, colour in zip(bplot['boxes'], colours_list):
             patch.set(facecolor=colour)

        for flier, colour in zip(bplot['fliers'], colours_list):
             flier.set(markeredgecolor=colour)


    if base_lines_pos is not None:
        xtickslocs = ax.get_xticks()
        for i in range(len(base_lines_pos)):

            if x_ticks_list[i] != base_lines_pos[i]:
                continue

            x_pos = xtickslocs[i]
            val = base_lines[i]
            ax.hlines(y=val, xmin=x_pos - 0.25, xmax=x_pos + 0.25, colors='r', linestyles='--', lw=1)

    ax.yaxis.grid(True)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig(save_path, dpi=300)
    plt.clf()


def get_baseline_EF_results_all(scores_folder):
    baseline_folder = os.path.join(scores_folder, 'Baseline')
    folders = os.listdir(baseline_folder)

    out_dict = {}

    ef_errors_lists, edv_errors_lists, esv_errors_lists = [], [], []
    ef_errors_mean, edv_errors_mean, esv_errors_mean = [], [], []
    ef_lists, edv_lists, esv_lists = [], [], []
    ef_mean, edv_mean, esv_mean = [], [], []
    percentages_train = []
    x_ticks = []

    for folder in folders:
        if not 'Baseline_with_' in folder:
            continue

        baseline_percentage_folder = os.path.join(baseline_folder, folder)

        labelled_dataset_counts = load_json(os.path.join(baseline_percentage_folder, 'labelled_dataset_counts.json'))
        percentage_train = labelled_dataset_counts['perc_train']
        num_train = labelled_dataset_counts['train']

        run_0_folder = os.path.join(baseline_percentage_folder, f'baseline_model_results_run_{0}')
        run_1_folder = os.path.join(baseline_percentage_folder, f'baseline_model_results_run_{1}')

        res_ef_file0 = os.path.join(run_0_folder, f'Consensus_EF_Vol_results_all.json')
        res_ef_file1 = os.path.join(run_1_folder, f'Consensus_EF_Vol_results_all.json')

        dict1 = load_json(res_ef_file0)
        dict2 = load_json(res_ef_file1)
        ef_err_list, edv_err_list, esv_err_list = [], [], []
        ef_list, edv_list, esv_list = [], [], []
        for key, value in dict1.items():
            value1 = dict2[key]

            if value['Ejection Fraction pred'] == 0 or value1['Ejection Fraction pred'] == 0:
                continue

            ef_err_list.append((value['EF error'] + value1['EF error']) / 2.0)
            edv_err_list.append((value['EDV error'] + value1['EDV error']) / 2.0)
            esv_err_list.append((value['ESV error'] + value1['ESV error']) / 2.0)
            ef_list.append((value['Ejection Fraction pred'] + value1['Ejection Fraction pred']) / 2.0)
            edv_list.append((value['EDV pred'] + value1['EDV pred']) / 2.0)
            esv_list.append((value['ESV pred'] + value1['ESV pred']) / 2.0)

        ef_errors_mean.append(np.mean(ef_err_list))
        edv_errors_mean.append(np.mean(edv_err_list))
        esv_errors_mean.append(np.mean(esv_err_list))

        ef_mean.append(np.mean(ef_list))
        edv_mean.append(np.mean(edv_list))
        esv_mean.append(np.mean(esv_list))

        ef_errors_lists.append(ef_err_list)
        edv_errors_lists.append(edv_err_list)
        esv_errors_lists.append(esv_err_list)

        ef_lists.append(ef_list)
        edv_lists.append(edv_list)
        esv_lists.append(esv_list)

        percentages_train.append(percentage_train)
        x_ticks.append(f'{num_train}({percentage_train}%)')

    if len(x_ticks) != 0:
        (percentages_train, x_ticks,
         ef_lists, edv_lists, esv_lists,
         ef_errors_lists, edv_errors_lists, esv_errors_lists,
         ef_errors_mean, edv_errors_mean, esv_errors_mean,
         ef_mean, edv_mean, esv_mean) = zip(
            *sorted(zip(percentages_train, x_ticks,
                        ef_lists, edv_lists, esv_lists,
                        ef_errors_lists, edv_errors_lists, esv_errors_lists,
                        ef_errors_mean, edv_errors_mean, esv_errors_mean,
                        ef_mean, edv_mean, esv_mean)))

    out_dict = {'percentages_train':percentages_train, 'x_ticks':x_ticks,
                'ef_lists':ef_lists, 'edv_lists':edv_lists, 'esv_lists':esv_lists,
                'ef_errors_lists':ef_errors_lists, 'edv_errors_lists':edv_errors_lists,
                'esv_errors_lists':esv_errors_lists,
                'ef_errors_mean':ef_errors_mean, 'edv_errors_mean':edv_errors_mean, 'esv_errors_mean':esv_errors_mean,
                'ef_mean':ef_mean, 'edv_mean':edv_mean, 'esv_mean':esv_mean}

    return out_dict


def create_plots_for_downstream_percentage_EF_comparison(main_folder):
    plt.clf()

    baseline_res_ef_dict = get_baseline_EF_results_all(os.path.relpath('.'))

    exp_folders = os.listdir(main_folder)
    exp_folders.sort()

    # Get results for each encoder at different percentages of downstream training data

    skip = 1

    results_dict = {}
    for i in range(len(exp_folders)):
        if 'exp_with' not in exp_folders[i].lower():
            continue

        exp_folder = os.path.join(main_folder, exp_folders[i])
        downstream_folders = os.listdir(exp_folder)

        ef_errors_lists, edv_errors_lists, esv_errors_lists = [],[],[]
        ef_lists, edv_lists, esv_lists = [],[],[]
        percentages_train = []
        x_ticks = []
        percentage_unlabelled = None

        for downstream_folder in downstream_folders:
            if 'finetuning_' not in downstream_folder.lower():
                continue

            results_folder = os.path.join(exp_folder, downstream_folder)
            if not os.path.exists(results_folder):
                continue

            files = os.listdir(results_folder)

            res_file = 'encoder_fine_tune_results.json'
            res_ef_file = f'Consensus_EF_Vol_results_all_{0}.json'
            res_ef_file_run_1 = f'Consensus_EF_Vol_results_all_{1}.json'
            if not (res_file in files or res_ef_file in files):
                continue

            res_dict = load_json(os.path.join(results_folder, res_file))

            percentage_unlabelled = res_dict['percentage_unlabelled']
            percentage_train = res_dict['percentage_train']
            num_train = res_dict['num_train']

            dict1 = load_json(os.path.join(results_folder, res_ef_file))
            dict2 = load_json(os.path.join(results_folder, res_ef_file_run_1))
            ef_err_list, edv_err_list, esv_err_list = [],[],[]
            ef_list, edv_list, esv_list = [], [], []
            for key, value in dict1.items():
                value1 = dict2[key]

                if value['Ejection Fraction pred']==0 or value1['Ejection Fraction pred']==0:
                    continue

                ef_err_list.append((value['EF error'] + value1['EF error'])/2.0)
                edv_err_list.append((value['EDV error'] + value1['EDV error'])/2.0)
                esv_err_list.append((value['ESV error'] + value1['ESV error'])/2.0)
                ef_list.append((value['Ejection Fraction pred'] + value1['Ejection Fraction pred'])/2.0)
                edv_list.append((value['EDV pred'] + value1['EDV pred'])/2.0)
                esv_list.append((value['ESV pred'] + value1['ESV pred'])/2.0)

            ef_errors_lists.append(ef_err_list)
            edv_errors_lists.append(edv_err_list)
            esv_errors_lists.append(esv_err_list)

            ef_lists.append(ef_list)
            edv_lists.append(edv_list)
            esv_lists.append(esv_list)

            percentages_train.append(percentage_train)
            x_ticks.append(f'{num_train}({percentage_train}%)')

        if len(x_ticks) != 0:
            percentages_train, x_ticks, ef_lists, edv_lists, esv_lists,  ef_errors_lists, edv_errors_lists, esv_errors_lists = zip(*sorted(zip(percentages_train, x_ticks,
                                                                           ef_lists, edv_lists, esv_lists,
                                                                           ef_errors_lists, edv_errors_lists, esv_errors_lists)))


            #Get baseline means to show on plots:
            base_ef_errors_mean = baseline_res_ef_dict['ef_errors_mean']
            base_edv_errors_mean = baseline_res_ef_dict['edv_errors_mean']
            base_esv_errors_mean = baseline_res_ef_dict['esv_errors_mean']
            base_ef_mean = baseline_res_ef_dict['ef_mean']
            base_edv_mean = baseline_res_ef_dict['edv_mean']
            base_esv_mean = baseline_res_ef_dict['esv_mean']
            base_percentages_train = baseline_res_ef_dict['percentages_train']

            create_box_plot(save_path=os.path.join(exp_folder, f'box_EF_error_{percentage_unlabelled}.png'),
                            data_lists=ef_errors_lists[skip:],
                            colours_list=[],
                            x_ticks_list=percentages_train[skip:],
                            x_label='Downstream Percentage',
                            y_label='EF Error',
                            base_lines=None,#base_ef_errors_mean[skip:],
                            base_lines_pos=None#base_percentages_train[skip:]
                            )

            create_box_plot(save_path=os.path.join(exp_folder, f'box_EDV_error_{percentage_unlabelled}.png'),
                            data_lists=edv_errors_lists[skip:],
                            colours_list=[],
                            x_ticks_list=percentages_train[skip:],
                            x_label='Downstream Percentage',
                            y_label='EDV Error')

            create_box_plot(save_path=os.path.join(exp_folder, f'box_ESV_error_{percentage_unlabelled}.png'),
                            data_lists=esv_errors_lists[skip:],
                            colours_list=[],
                            x_ticks_list=percentages_train[skip:],
                            x_label='Downstream Percentage',
                            y_label='ESV Error')


def write_list_to_json(list_data, folder, file_name, key):
    dict1 = {key:list_data}
    file_name = os.path.join(folder, file_name)
    with open(file_name, 'w') as fp:
        json.dump(dict1, fp)

def write_dict_to_json(dict_data, folder, file_name):
    file_name = os.path.join(folder, file_name)
    with open(file_name, 'w') as fp:
        json.dump(dict_data, fp)

def load_list_from_json(json_file, key):
    dict1 = None
    with open(json_file, 'r') as fp:
        dict1 = json.load(fp)
    return dict1[key]

def load_json(json_file):
    dict1 = None
    with open(json_file, 'r') as fp:
        dict1 = json.load(fp)
    return dict1


def get_list_of_images_and_labels_from_folder(images_path, images_folder_name, labels_folder_name, add_png_ext=True):
    # The object just has names, so lets append path

    if images_path == None:
        return [], []

    images_folder = os.path.join(images_path, images_folder_name)
    labels_folder = os.path.join(images_path, labels_folder_name)
    images = []
    labels = []

    label_files = os.listdir(labels_folder)

    for file in label_files:
        if (add_png_ext):
            image_file = os.path.join(images_folder, file + '.png')
        else:
            image_file = os.path.join(images_folder, file)
        label_file = os.path.join(labels_folder, file)

        if not os.path.exists(image_file):
            continue
        if not os.path.exists(label_file):
            continue

        images.append(image_file)
        labels.append(label_file)

    return images, labels

import concurrent.futures
from multiprocessing import Pool

def compute_mask_volume_single(model_output):

    mask = np.array(model_output)
    mask = mask[:, :, 0]
    mask = tf.greater(mask, 0.5)
    mask = tf.dtypes.cast(mask, tf.uint8)
    mask = mask.numpy()

    (vol, poly_points, minmaxline,
     midpointline, segments) = get_mask_volume_quick(mask, K=20, is_binary_image=True)

    if poly_points is not None:
        poly_points = poly_points.tolist()

    return (vol, poly_points, minmaxline, midpointline, segments)

def compute_mask_volume_from_json_parallel(file):
    vols_dict = {}

    raw_output_dict = load_json(file)

    frames_list = raw_output_dict['frames_list']
    raw_outputs = raw_output_dict['raw_outputs']

    #         vols_dict[os.path.basename(name)] = vals
    with Pool(64) as pool:
        for name, vals in zip(frames_list, pool.map(compute_mask_volume_single, raw_outputs)):
                vols_dict[os.path.basename(name)] = vals

    return vols_dict

def compute_mask_volume_from_json(file):

    vols_dict = {}

    raw_output_dict = load_json(file)

    frames_list = raw_output_dict['frames_list']
    raw_outputs = raw_output_dict['raw_outputs']

    for i in range(len(frames_list)):
        name = frames_list[i]
        model_output = raw_outputs[i]

        mask = np.array(model_output)
        mask = mask[:, :, 0]
        mask = tf.greater(mask, 0.5)
        mask = tf.dtypes.cast(mask, tf.uint8)
        mask = mask.numpy()

        (vol, poly_points, minmaxline,
         midpointline, segments) = get_mask_volume_quick(mask, K=20, is_binary_image=True)

        if poly_points is not None:
            poly_points = poly_points.tolist()

        vols_dict[os.path.basename(name)] = (vol, poly_points, minmaxline,
                                            midpointline, segments)

    return vols_dict

def compute_mask_volume_from_labels_folder(path):
    files = os.listdir(path)

    vols_dict = {}

    for file in files:
        mask = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)

        (vol, poly_points, minmaxline,
         midpointline, segments) = get_mask_volume_quick(mask, K=20, is_binary_image=True)

        if poly_points is not None:
            poly_points = poly_points.tolist()

        vols_dict[file] = [vol, poly_points, minmaxline, midpointline, segments]

    return vols_dict

def annotate_mask_disks(save_folder, consensus_images_path, vols_dict):

    files = os.listdir(consensus_images_path)

    for file in files:

        img = cv2.imread(os.path.join(consensus_images_path, file), cv2.IMREAD_GRAYSCALE)
        (vol, poly_points, minmaxline,
         midpointline, segments) = vols_dict[file]
        if poly_points is not None:
            img = annotate_image(cv2.cvtColor(img,cv2.COLOR_GRAY2RGB), poly_points, None, midpointline, segments, is_binary_image=False)
        cv2.imwrite(os.path.join(save_folder, file), img)

def get_consensus_ratios():
    ratios_dict = {}
    with open('consensus_ratios.txt') as f:
        lines = f.readlines()
    for line in lines:
        vals = line.split()
        file_name = vals[0].replace('\'', '')
        file_name = file_name[0: file_name.rindex('-')]
        ratios_dict[file_name] = float(vals[1])

    return ratios_dict

def calculate_consensus_ground_truth_volumes(consensus_dataset_path):
    vols_gt_dict = {}
    gt_file = os.path.join('.', 'consensus_ground_truth_volumes.json')
    if os.path.exists(gt_file):
        vols_gt_dict = load_json(gt_file)
    else:
        vols_gt_dict = compute_mask_volume_from_labels_folder(os.path.join(consensus_dataset_path, 'Labels'))
        write_dict_to_json(vols_gt_dict, '.', 'consensus_ground_truth_volumes.json')
        if not os.path.exists(os.path.join('.', 'consensus_disks_volumes')):
            os.makedirs(os.path.join('.', 'consensus_disks_volumes'))
        annotate_mask_disks(os.path.join('.', 'consensus_disks_volumes'),
                            os.path.join(consensus_dataset_path, 'Images'),
                            vols_gt_dict)

    return vols_gt_dict

def compute_ejection_fraction_consensus_set(main_folder, consensus_dataset_path):
    #Get the pixel ratios
    ratios_dict = get_consensus_ratios()

    #Calculate and save consensus ground truth volumes:
    vols_gt_dict = calculate_consensus_ground_truth_volumes(consensus_dataset_path)

    vols_ml_gt_dict = convert_consensus_vols_to_ml(vols_gt_dict, ratios_dict)

    #Calculate the volumes from the raw model output predictions
    exp_folders = os.listdir(main_folder)
    exp_folders.sort()

    for i in range(len(exp_folders)):
        if 'exp_with' not in exp_folders[i].lower():
            continue

        exp_folder = os.path.join(main_folder, exp_folders[i])
        downstream_folders = os.listdir(exp_folder)
        unlabelled_perc_str = exp_folders[i].lower().replace('exp_with_', '')
        # Careful if percentage is a float but I doubt I will use a float percentage for unlabelled data.
        unlabelled_perc = float(unlabelled_perc_str.replace('_percent_unlabelled_data',''))

        for downstream_folder in downstream_folders:
            if 'finetuning_' not in downstream_folder.lower():
                continue

            results_folder = os.path.join(exp_folder, downstream_folder)
            if not os.path.exists(results_folder):
                continue

            files = os.listdir(results_folder)
            avg_ef_error_list, avg_ED_vol_error_list, avg_ES_vol_error_list, count_no_ef_list = [],[],[],[]
            for file in files:
                if not ('raw_model_outputs.json' in file.lower() and 'consensus' in file.lower()):
                    continue

                pos = file.rindex('_raw_model_outputs')
                run_number = int(file[pos - 1:pos])  # -1 assuming the run number is only a single digit

                disks_file = os.path.join(results_folder, f'Consensus_disks_volumes_run_{run_number}.json')

                #compare date of raw model output file file and disk file
                recreate_disk_file = False
                if os.path.exists(disks_file):
                    model_output_date = os.path.getmtime(os.path.join(results_folder, file))
                    disk_file_date = os.path.getmtime(disks_file)
                    if model_output_date >  disk_file_date:
                        recreate_disk_file = True
                        print(f'** RE-CREATING FILE: {disks_file}')

                if os.path.exists(disks_file) and recreate_disk_file==False:
                   vols_dict = load_json(disks_file)
                else:
                    vols_dict = compute_mask_volume_from_json(os.path.join(results_folder, file))
                    write_dict_to_json(vols_dict,
                                       results_folder,
                                       f'Consensus_disks_volumes_run_{run_number}.json')
                    save_disks_folder = os.path.join(results_folder, f'Consensus_disks_volumes_run_{run_number}')
                    if not os.path.exists(save_disks_folder):
                        os.makedirs(save_disks_folder)
                    annotate_mask_disks(save_disks_folder,
                                        os.path.join(consensus_dataset_path, 'Images'),
                                        vols_dict)

                #calc ejection fractions but first convert pixel vols to cm3/ml
                vols_ml_dict = convert_consensus_vols_to_ml(vols_dict, ratios_dict)
                (info_dict,
                 info_dict_list,
                 avg_ef_error,
                 avg_ED_vol_error,
                 avg_ES_vol_error,
                 count_no_ef) = calc_ejection_fractions_and_errors(vols_ml_gt_dict, vols_ml_dict)
                write_dict_to_json(info_dict,
                                   results_folder,
                                   f'Consensus_EF_Vol_results_all_{run_number}.json')
                #Also save above json as csv
                df = pd.DataFrame(info_dict_list)
                df.to_csv(os.path.join(results_folder, f'Consensus_EF_Vol_results_all_{run_number}.csv'), index=False)
                avg_ef_error_list.append(avg_ef_error)
                avg_ED_vol_error_list.append(avg_ED_vol_error)
                avg_ES_vol_error_list.append(avg_ES_vol_error)
                count_no_ef_list.append(count_no_ef)

            write_dict_to_json({'avg_ef_errors_per_run':avg_ef_error_list,
                                'avg_ed_vol_errors_per_run':avg_ED_vol_error_list,
                                'avg_es_vol_errors_per_run':avg_ES_vol_error_list,
                                'count_no_ef':count_no_ef_list,
                                'percentage_unlabelled': unlabelled_perc},
                               results_folder,
                               f'Consensus_EF_Vol_error_results.json')

def compute_ejection_fraction_consensus_set_for_baseline(folder, consensus_dataset_path):
    #Get the pixel ratios
    ratios_dict = get_consensus_ratios()

    #Calculate and save consensus ground truth volumes:
    vols_gt_dict = calculate_consensus_ground_truth_volumes(consensus_dataset_path)

    vols_ml_gt_dict = convert_consensus_vols_to_ml(vols_gt_dict, ratios_dict)

    #Calculate the volumes from the raw model output predictions

    files = os.listdir(folder)
    avg_ef_error, avg_ED_vol_error, avg_ES_vol_error = None, None, None
    for file in files:
        if not ('raw_model_outputs.json' in file.lower() and 'consensus' in file.lower()):
            continue

        pos = file.rindex('_raw_model_outputs')

        disks_file = os.path.join(folder, f'Consensus_disks_volumes.json')

        # # compare date of raw model output file file and disk file
        # if model_output_date > disk_file_date:

        # compare date of raw model output file and disk file
        recreate_disk_file = False
        if os.path.exists(disks_file):
            model_output_date = os.path.getmtime(os.path.join(folder, file))
            disk_file_date = os.path.getmtime(disks_file)
            if model_output_date > disk_file_date:
                recreate_disk_file = True
                print(f'** RE-CREATING FILE: {disks_file}')

        if os.path.exists(disks_file) and recreate_disk_file == False:
           vols_dict = load_json(disks_file)
        else:
            vols_dict = compute_mask_volume_from_json(os.path.join(folder, file))
            write_dict_to_json(vols_dict,
                               folder,
                               f'Consensus_disks_volumes.json')
            save_disks_folder = os.path.join(folder, f'Consensus_disks_volumes')
            if not os.path.exists(save_disks_folder):
                os.makedirs(save_disks_folder)
            annotate_mask_disks(save_disks_folder,
                                os.path.join(consensus_dataset_path, 'Images'),
                                vols_dict)

        #calc ejection fractions but first convert pixel vols to cm3/ml
        vols_ml_dict = convert_consensus_vols_to_ml(vols_dict, ratios_dict)
        (info_dict,
         info_dict_list,
         avg_ef_error,
         avg_ED_vol_error,
         avg_ES_vol_error,
         count_no_ef) = calc_ejection_fractions_and_errors(vols_ml_gt_dict, vols_ml_dict)
        write_dict_to_json(info_dict,
                           folder,
                           f'Consensus_EF_Vol_results_all.json')
        #Also save above json as csv
        df = pd.DataFrame(info_dict_list)
        df.to_csv(os.path.join(folder, f'Consensus_EF_Vol_results_all.csv'), index=False)

    write_dict_to_json({'avg_ef_error':avg_ef_error,
                        'avg_ed_vol_error':avg_ED_vol_error,
                        'avg_es_vol_error':avg_ES_vol_error},
                       folder,
                       f'Consensus_EF_Vol_error_results.json')

    return avg_ef_error, avg_ED_vol_error, avg_ES_vol_error, count_no_ef

def calc_ejection_fractions_and_errors(vols_ml_gt_dict, vols_ml_dict):

    info_dict = {}
    info_dict_list = []
    avg_ef_error, avg_ED_vol_error, avg_ES_vol_error = 0, 0, 0

    max_EF_error, max_ed_vol_error, max_es_vol_error = 0,0,0
    mean_EF_error, mean_ed_vol_error, mean_es_vol_error = 0, 0, 0
    for key, value in vols_ml_gt_dict.items():
        gt_vols_list = list(value.values())
        gt_frames_list = list(value.keys())
        pred_vols_dict = vols_ml_dict[key]
        pred_vols_list = list(pred_vols_dict.values())
        pred_frames_list = list(pred_vols_dict.keys())

        gt_ED_frame, gt_ES_frame = gt_frames_list[0], gt_frames_list[1]
        gt_ED_vol, gt_ES_vol = gt_vols_list[0], gt_vols_list[1]

        if gt_vols_list[1] > gt_vols_list[0]:
            gt_ED_frame = gt_frames_list[1]
            gt_ES_frame = gt_frames_list[0]
            gt_ED_vol = gt_vols_list[1]
            gt_ES_vol = gt_vols_list[0]

        pred_ED_frame, pred_ES_frame = gt_ED_frame, gt_ES_frame
        pred_ED_vol, pred_ES_vol = pred_vols_dict[gt_ED_frame], pred_vols_dict[gt_ES_frame]

        eject_frac_gt, strk_vol_gt = calc_ejection_fraction(gt_ED_vol, gt_ES_vol)
        eject_frac_pred, strk_vol_pred = 0, 0
        if not (pred_ED_vol == 0 or pred_ES_vol == 0) and not pred_ED_vol < pred_ES_vol:
            eject_frac_pred, strk_vol_pred = calc_ejection_fraction(pred_ED_vol, pred_ES_vol)
            ef_error = abs(eject_frac_gt - eject_frac_pred)
            ED_vol_error = abs(gt_ED_vol - pred_ED_vol)
            ES_vol_error = abs(gt_ES_vol - pred_ES_vol)

            if ef_error > max_EF_error: max_EF_error = ef_error
            if ED_vol_error > max_ed_vol_error: max_ed_vol_error = ED_vol_error
            if ES_vol_error > max_es_vol_error: max_es_vol_error = ES_vol_error

            mean_EF_error+=ef_error
            mean_ed_vol_error+=ED_vol_error
            mean_es_vol_error+=ES_vol_error

    mean_EF_error/= len(vols_ml_gt_dict)
    mean_ed_vol_error /= len(vols_ml_gt_dict)
    mean_es_vol_error /= len(vols_ml_gt_dict)

    count_no_ef = 0
    for key, value in vols_ml_gt_dict.items():
        gt_vols_list = list(value.values())
        gt_frames_list = list(value.keys())
        pred_vols_dict = vols_ml_dict[key]
        pred_vols_list = list(pred_vols_dict.values())
        pred_frames_list = list(pred_vols_dict.keys())

        gt_ED_frame, gt_ES_frame = gt_frames_list[0], gt_frames_list[1]
        gt_ED_vol, gt_ES_vol = gt_vols_list[0], gt_vols_list[1]

        if gt_vols_list[1] > gt_vols_list[0]:
            gt_ED_frame = gt_frames_list[1]
            gt_ES_frame = gt_frames_list[0]
            gt_ED_vol = gt_vols_list[1]
            gt_ES_vol = gt_vols_list[0]

        # We do it this way because we want to match the ED & ES frames with the respective ground truths.
        pred_ED_frame, pred_ES_frame = gt_ED_frame, gt_ES_frame
        pred_ED_vol, pred_ES_vol = pred_vols_dict[gt_ED_frame], pred_vols_dict[gt_ES_frame]

        eject_frac_gt, strk_vol_gt = calc_ejection_fraction(gt_ED_vol, gt_ES_vol)
        eject_frac_pred, strk_vol_pred = 0, 0
        if not (pred_ED_vol == 0 or pred_ES_vol == 0) and not pred_ED_vol < pred_ES_vol:
            eject_frac_pred, strk_vol_pred = calc_ejection_fraction(pred_ED_vol, pred_ES_vol)
            ef_error = abs(eject_frac_gt - eject_frac_pred)
            ED_vol_error = abs(gt_ED_vol - pred_ED_vol)
            ES_vol_error = abs(gt_ES_vol - pred_ES_vol)
        else:
            #continue
            count_no_ef+=1
            ef_error = 0#mean_EF_error
            ED_vol_error = 0#mean_ed_vol_error
            ES_vol_error = 0#mean_es_vol_error


        info_dict[key] = {'Ejection Fraction gt':eject_frac_gt,
                          'Ejection Fraction pred':eject_frac_pred,
                          'EF error': ef_error,
                          'EDV gt': gt_ED_vol,
                          'ESV gt': gt_ES_vol,
                          'EDV pred': pred_ED_vol,
                          'ESV pred': pred_ES_vol,
                          'EDV error': ED_vol_error,
                          'ESV error': ES_vol_error,
                          'ED frame': pred_ED_frame,
                          'ES frame': pred_ES_frame
                          }
        info_dict_list.append({'Ejection Fraction gt':eject_frac_gt,
                          'Ejection Fraction pred':eject_frac_pred,
                          'EF error': ef_error,
                          'EDV gt': gt_ED_vol,
                          'ESV gt': gt_ES_vol,
                          'EDV pred': pred_ED_vol,
                          'ESV pred': pred_ES_vol,
                          'EDV error': ED_vol_error,
                          'ESV error': ES_vol_error,
                          'ED frame': pred_ED_frame,
                          'ES frame': pred_ES_frame
                          })
        avg_ef_error += ef_error
        avg_ED_vol_error += ED_vol_error
        avg_ES_vol_error += ES_vol_error

    if count_no_ef < len(vols_ml_gt_dict):
        avg_ef_error /= (len(vols_ml_gt_dict) - count_no_ef)
        avg_ED_vol_error /= (len(vols_ml_gt_dict) - count_no_ef)
        avg_ES_vol_error /= (len(vols_ml_gt_dict) - count_no_ef)
    else:
        avg_ef_error = -1
        avg_ED_vol_error = -1
        avg_ES_vol_error = -1

    return info_dict, info_dict_list, avg_ef_error, avg_ED_vol_error, avg_ES_vol_error, count_no_ef


def convert_consensus_vols_to_ml(vols_dict, ratios_dict):
    vols_ml_dict = {}

    for key, value in vols_dict.items():
        frame = key
        (vol, poly_points, minmaxline,
         midpointline, segments) = value
        frame_key = frame[0:frame.rindex('-')]
        ratio = ratios_dict[frame_key]
        vol_ml = vol * math.pow(ratio, 3)
        if frame_key in vols_ml_dict:
            pair_dict = vols_ml_dict[frame_key]
            pair_dict[frame] = vol_ml
        else:
            vols_ml_dict[frame_key] = {frame: vol_ml}

    return vols_ml_dict


def calc_ejection_fraction(EDV, ESV):

    stroke_volume = abs(EDV - ESV)
    ejection_fraction = (stroke_volume / EDV) * 100.0

    return ejection_fraction, stroke_volume
