import pandas as pd
from sklearn import metrics
from os import listdir
from os.path import isfile, join

from FoldMetrics import FoldMetrics
from PrCurve import PrCurve
from RocCurve import RocCurve


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

        df.columns = ["true_label", "pred"]
        data_frames.append(df)

    return data_frames


def calculate_metrics(data_frames):
    model_metrics = []

    for df in data_frames:
        fpr, tpr, roc_thresholds = metrics.roc_curve(df['true_label'], df['pred'])
        precision, recall, prc_thresholds = metrics.precision_recall_curve(df['true_label'], df['pred'])

        model_metrics.append(FoldMetrics(
            RocCurve(fpr, tpr, roc_thresholds),
            PrCurve(precision, recall, prc_thresholds),
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


stream_pred_path = ''
batch_pred_path = ''
output_path = ''

print('Retrieving files names from directories...')

stream_file_names = get_files_names(stream_pred_path)
batch_files_names = get_files_names(batch_pred_path)

print('Getting data frames from files...')

batch_dfs = get_dfs_from_files(batch_pred_path, batch_files_names, False, None)
stream_dfs = get_dfs_from_files(stream_pred_path, stream_file_names, True, len(batch_dfs[0]))

print('Calculating each fold metrics...')

stream_metrics = calculate_metrics(stream_dfs)
batch_metrics = calculate_metrics(batch_dfs)
batch_dfs = []
stream_dfs = []

print('Creating data frames with metrics for each algorithm...')

stream_aps, stream_rocaucs = get_separate_metrics_arrays(stream_metrics)
batch_aps, batch_rocaucs = get_separate_metrics_arrays(stream_metrics)

stream_metrics_df = pd.DataFrame(data={'avg_prec': stream_aps, 'roc_auc': stream_rocaucs})
batch_metrics_df = pd.DataFrame(data={'avg_prec': batch_aps, 'roc_auc': batch_rocaucs})

roc_curve_df = pd.DataFrame(data={'stream_fpr': stream_metrics[0].roc_curve.fpr,
                                  'stream_tpr': stream_metrics[0].roc_curve.tpr,
                                  'stream_thresholds': stream_metrics[0].roc_curve.thresholds,
                                  'batch_fpr': batch_metrics[0].roc_curve.fpr,
                                  'batch_tpr': batch_metrics[0].roc_curve.tpr,
                                  'batch_thresholds': batch_metrics[0].roc_curve.thresholds})

pr_curve_df = pd.DataFrame(data={'stream_precision': stream_metrics[0].pr_curve.precision,
                                 'stream_recall': stream_metrics[0].pr_curve.recall,
                                 'stream_thresholds': stream_metrics[0].pr_curve.thresholds,
                                 'batch_precision': batch_metrics[0].pr_curve.precision,
                                 'batch_recall': batch_metrics[0].pr_curve.recall,
                                 'batch_thresholds': batch_metrics[0].pr_curve.thresholds})

print('Saving to files...')

stream_metrics_df.to_csv(join(output_path, 'stream_metrics_1.csv'), index=False)
batch_metrics_df.to_csv(join(output_path, 'batch_metrics_1.csv'), index=False)
roc_curve_df.to_csv(join(output_path, 'roc_curve_1.csv'), index=False)
pr_curve_df.to_csv(join(output_path, 'pr_curve_1.csv'), index=False)
