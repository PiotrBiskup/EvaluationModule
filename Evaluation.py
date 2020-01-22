import pandas as pd
from sklearn import metrics
from os import listdir
from os.path import isfile, join

from FoldMetrics import FoldMetrics
from PrCurve import PrCurve
from RocCurve import RocCurve

output_path = '/Users/pbiskup/Documents/Inz/pred/output'
output_roc_curve_path = '/Users/pbiskup/Documents/Inz/pred/output/curves/roc'
output_pr_curve_path = '/Users/pbiskup/Documents/Inz/pred/output/curves/pr'

num = 9  # number of ad campaign 1-9

stream_pred_path = '/Users/pbiskup/Documents/Inz/pred/stream/s' + str(num) + '/'
batch_pred_path = '/Users/pbiskup/Documents/Inz/pred/batch/b' + str(num) + '/'
sm_file_name = 'stream_metrics_' + str(num) + '.csv'
bm_file_name = 'batch_metrics_' + str(num) + '.csv'
stream_rocc_file_name = 'stream_roc_curve_' + str(num) + '.csv'
batch_rocc_file_name = 'batch_roc_curve_' + str(num) + '.csv'
stream_prc_file_name = 'stream_pr_curve_' + str(num) + '.csv'
batch_prc_file_name = 'batch_pr_curve_' + str(num) + '.csv'


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
batch_dfs = []
stream_dfs = []

print('Creating data frames with metrics for each algorithm...')

stream_aps, stream_rocaucs = get_separate_metrics_arrays(stream_metrics)
batch_aps, batch_rocaucs = get_separate_metrics_arrays(stream_metrics)

stream_metrics_df = pd.DataFrame(data={'avg_prec': stream_aps, 'roc_auc': stream_rocaucs})
batch_metrics_df = pd.DataFrame(data={'avg_prec': batch_aps, 'roc_auc': batch_rocaucs})

stream_roc_curve_df = pd.DataFrame(data={'stream_fpr': stream_metrics[0].roc_curve.fpr,
                                         'stream_tpr': stream_metrics[0].roc_curve.tpr})

batch_roc_curve_df = pd.DataFrame(data={'batch_fpr': batch_metrics[0].roc_curve.fpr,
                                        'batch_tpr': batch_metrics[0].roc_curve.tpr})

stream_pr_curve_df = pd.DataFrame(data={'stream_precision': stream_metrics[0].pr_curve.precision,
                                        'stream_recall': stream_metrics[0].pr_curve.recall})

batch_pr_curve_df = pd.DataFrame(data={'batch_precision': batch_metrics[0].pr_curve.precision,
                                       'batch_recall': batch_metrics[0].pr_curve.recall})

print('Saving to files...')

stream_metrics_df.to_csv(join(output_path, sm_file_name), index=False)
batch_metrics_df.to_csv(join(output_path, bm_file_name), index=False)
stream_roc_curve_df.to_csv(join(output_roc_curve_path, stream_rocc_file_name), index=False)
batch_roc_curve_df.to_csv(join(output_roc_curve_path, batch_rocc_file_name), index=False)
stream_pr_curve_df.to_csv(join(output_pr_curve_path, stream_prc_file_name), index=False)
batch_pr_curve_df.to_csv(join(output_pr_curve_path, batch_prc_file_name), index=False)
