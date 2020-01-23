import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from os import listdir
from os.path import isfile, join

from FoldMetrics import FoldMetrics
from PrCurve import PrCurve
from RocCurve import RocCurve

output_path = '/Users/pbiskup/Documents/Inz/pred/output'
output_roc_curve_path = '/Users/pbiskup/Documents/Inz/pred/output/curves/roc'
output_pr_curve_path = '/Users/pbiskup/Documents/Inz/pred/output/curves/pr'


def get_files_names(dir_path):
    return [file for file in listdir(dir_path) if isfile(join(dir_path, file))]


def get_dfs_from_files(dir_path, files_names, only_tail, tail_len):
    data_frames = []

    for file in files_names:
        path = join(dir_path, file)

        if only_tail:
            df = pd.read_csv(path, sep=" ", header=None).tail(tail_len)
        else:
            df = pd.read_csv(path, sep=" ", header=None)

        data_frames.append(df)

    return data_frames


def get_dfs_from_files_with_headers(dir_path, files_names, only_tail=False, tail_len=None):
    data_frames = []

    for file in files_names:
        path = join(dir_path, file)

        if only_tail:
            df = pd.read_csv(path, sep=",").tail(tail_len)
        else:
            df = pd.read_csv(path, sep=",")

        data_frames.append(df)

    return data_frames


def calculate_metrics(data_frames):
    model_metrics = []

    for df in data_frames:
        df.columns = ["true_label", "pred"]
        fpr, tpr, roc_thresholds = metrics.roc_curve(df['true_label'], df['pred'])
        precision, recall, prc_thresholds = metrics.precision_recall_curve(df['true_label'], df['pred'])

        model_metrics.append(FoldMetrics(
            RocCurve(fpr, tpr),
            PrCurve(precision, recall),
            metrics.average_precision_score(df['true_label'], df['pred'])
        ))

    return model_metrics


def get_separate_metrics_arrays(folds_metrics_array):
    avg_prec_array = []
    roc_auc_array = []
    for fold in folds_metrics_array:
        avg_prec_array.append(fold.avg_prec)
        roc_auc_array.append(fold.roc_curve.roc_auc)
    return avg_prec_array, roc_auc_array


def process_pred_files(ad_num):
    stream_pred_path = '/Users/pbiskup/Documents/Inz/pred/stream/s' + str(ad_num) + '/'
    batch_pred_path = '/Users/pbiskup/Documents/Inz/pred/batch/b' + str(ad_num) + '/'
    sm_file_name = 'stream_metrics_' + str(ad_num) + '.csv'
    bm_file_name = 'batch_metrics_' + str(ad_num) + '.csv'
    roc_curve_name = 'roc_curve_' + str(ad_num) + '.png'
    pr_curve_name = 'pr_curve_' + str(ad_num) + '.png'

    print('Retrieving files names from directories...')

    stream_file_names = get_files_names(stream_pred_path)
    batch_files_names = get_files_names(batch_pred_path)
    stream_file_names.sort()
    batch_files_names.sort()

    print('Getting data frames from files...')

    batch_dfs = get_dfs_from_files(batch_pred_path, batch_files_names, False, None)
    stream_dfs = get_dfs_from_files(stream_pred_path, stream_file_names, True, len(batch_dfs[0]))

    print('Calculating each fold metrics...')

    stream_metrics = calculate_metrics(stream_dfs)
    batch_metrics = calculate_metrics(batch_dfs)

    print('Creating data frames with metrics for each algorithm...')

    stream_aps, stream_rocaucs = get_separate_metrics_arrays(stream_metrics)
    batch_aps, batch_rocaucs = get_separate_metrics_arrays(batch_metrics)

    stream_metrics_df = pd.DataFrame(data={'avg_prec': stream_aps, 'roc_auc': stream_rocaucs})
    batch_metrics_df = pd.DataFrame(data={'avg_prec': batch_aps, 'roc_auc': batch_rocaucs})

    random_pred_df = batch_dfs[0].copy()
    random_pred_df['pred'] = np.random.uniform(0, 1, random_pred_df.shape[0])
    random_fpr, random_tpr, random_roc_thresholds = metrics.roc_curve(random_pred_df['true_label'],
                                                                      random_pred_df['pred'])
    random_precision, random_recall, random_prc_thresholds = metrics.precision_recall_curve(
        random_pred_df['true_label'], random_pred_df['pred'])
    random_avg_prec = metrics.average_precision_score(random_pred_df['true_label'], random_pred_df['pred'])

    stream_metrics_df.to_csv(join(output_path, sm_file_name), index=False)
    batch_metrics_df.to_csv(join(output_path, bm_file_name), index=False)

    #  roc curve

    plt.plot(stream_metrics[0].roc_curve.fpr, stream_metrics[0].roc_curve.tpr,
             label='stream ROC curve (area = %0.3f)' % stream_metrics[0].roc_curve.roc_auc)

    plt.plot(batch_metrics[0].roc_curve.fpr, batch_metrics[0].roc_curve.tpr,
             label='batch ROC curve (area = %0.3f)' % batch_metrics[0].roc_curve.roc_auc)

    plt.plot(random_fpr, random_tpr,
             label='random ROC curve (area = %0.3f)' % metrics.auc(random_fpr, random_tpr))

    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    plt.savefig(join(output_path, roc_curve_name))
    plt.clf()

    #  precision recall curve

    y_limit = 1.0
    if ad_num == 1:
        y_limit = 0.6
    elif ad_num == 2:
        y_limit = 0.007
    elif ad_num == 3:
        y_limit = 0.004
    elif ad_num == 4:
        y_limit = 0.0035
    elif ad_num == 5:
        y_limit = 0.06
    elif ad_num == 6:
        y_limit = 0.6
    elif ad_num == 7:
        y_limit = 0.1
    elif ad_num == 8:
        y_limit = 0.6
    elif ad_num == 9:
        y_limit = 0.007

    plt.plot(stream_metrics[0].pr_curve.recall, stream_metrics[0].pr_curve.precision, marker='.',
             label='stream PR curve (avg prec = %0.3f)' % stream_metrics[0].avg_prec, zorder=5, linewidth=1,
             markersize=1)

    plt.plot(batch_metrics[0].pr_curve.recall, batch_metrics[0].pr_curve.precision, marker='.',
             label='batch PR curve (avg prec = %0.3f)' % batch_metrics[0].avg_prec, zorder=5, linewidth=1, markersize=1)

    plt.plot(random_recall, random_precision, marker='.',
             label='random PR curve (avg prec = %0.3f)' % random_avg_prec, zorder=5, linewidth=1, markersize=1)

    plt.ylim([0.0, y_limit])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="upper right")
    plt.title('Precision Recall Curve')
    plt.savefig(join(output_path, pr_curve_name))
    plt.clf()


def filter_files_with_name(files_names, contained_word):
    return [file_name for file_name in files_names if contained_word in file_name]


def concat_data_frames(data_frames):
    return pd.concat(data_frames, ignore_index=True, sort=False)


def combine_metrics_files():
    metrics_files_names = get_files_names(output_path)

    stream_metrics_file_names = filter_files_with_name(metrics_files_names, 'stream')
    batch_metrics_file_names = filter_files_with_name(metrics_files_names, 'batch')

    stream_metrics_file_names.sort()
    batch_metrics_file_names.sort()

    stream_metrics_dfs = get_dfs_from_files_with_headers(output_path, stream_metrics_file_names)
    batch_metrics_dfs = get_dfs_from_files_with_headers(output_path, batch_metrics_file_names)

    stream_df = concat_data_frames(stream_metrics_dfs)
    batch_df = concat_data_frames(batch_metrics_dfs)

    result_df = pd.DataFrame(data={
        'batch_roc_auc': batch_df['roc_auc'],
        'stream_roc_auc': stream_df['roc_auc'],
        'batch_avg_prec': batch_df['avg_prec'],
        'stream_avg_prec': stream_df['avg_prec']
    })

    result_df.to_csv(join(output_path, 'metrics.csv'), index=False)


for x in range(1, 10):
    process_pred_files(x)

combine_metrics_files()
