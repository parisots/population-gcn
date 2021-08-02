import sklearn
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from functools import partial
from multiprocessing import Pool
from main_ABIDE import train_fold_thread


def compute_error_and_bias(y_true, y_pred, a, absolute_value=True):
    # https://github.com/matthklein/equalized_odds_under_perturbation/blob
    # /master/equalized_odds.py
    # formula before Assumption 1 https://arxiv.org/pdf/1906.03284.pdf
    """
    Computes the error and the bias of a predictor.

    INPUT:
      y_true ...  in {-1,1}^n1; true labels
      y_pred ... in {-1,1}^n1; predicted labels
      a ... in {0,1}^n1; protected attributes
      
    OUTPUT:
      error ... error of the predictor
      biasY1 ... bias of the predictor for Y=1
      biasYm1 ... bias of the predictor for Y=-1
    """

    error = np.mean(np.not_equal(y_true, y_pred))
    # a=0 is female, a=1 is male
    true_positive_female = np.sum(np.logical_and(y_pred == 1,
                                                 np.logical_and(y_true == 1,
                                                                a == 0))) / \
                           float(
                               np.sum(
                                   np.logical_and(y_true == 1, a == 0)))
    true_positive_male = np.sum(np.logical_and(y_pred == 1,
                                               np.logical_and(y_true == 1,
                                                              a == 1))) / \
                         float(
                             np.sum(
                                 np.logical_and(y_true == 1, a == 1)))
    false_positive_female = np.sum(np.logical_and(y_pred == 1,
                                                  np.logical_and(y_true == 0,
                                                                 a == 0))) / \
                            float(
                                np.sum(
                                    np.logical_and(y_true == 0, a == 0)))
    false_positive_male = np.sum(np.logical_and(y_pred == 1,
                                                np.logical_and(y_true == 0,
                                                               a == 1))) / \
                          float(
                              np.sum(
                                  np.logical_and(y_true == 0, a == 1)))

    true_positive_bias = true_positive_female - true_positive_male
    false_positive_bias = false_positive_female - false_positive_male

    if absolute_value:
        true_positive_bias = np.abs(true_positive_bias)
        false_positive_bias = np.abs(false_positive_bias)

    return error, true_positive_bias, false_positive_bias


def get_roc_auc_safe(true, predicted):
    try:
        auc = sklearn.metrics.roc_auc_score(true, predicted)
    except ValueError:
        auc = np.nan
    return auc


def get_acc_safe(true, predicted):
    try:
        acc = sklearn.metrics.accuracy_score(true, predicted)
    except IndexError:
        acc = np.nan
    return acc


def get_auc_acc(y, pred, test, sex_data, population='all'):
    """
    population = 'all' / 'male' / 'female'
    """
    test_y = y[test, :] - 1
    pred_test_y = pred[test, :]
    pred_test_y_bin = (pred_test_y[:, 1] >= 0.5).astype(int)
    test_sex_data = sex_data[test, :]
    scores_dict = {}

    if population == 'all':
        auc_test = get_roc_auc_safe(test_y, pred_test_y[:, 1])
        acc_test = get_acc_safe(test_y, pred_test_y_bin)
        n = test.shape[0]

        # Accuracy per DX_GROUP
        acc_test_asd = get_acc_safe(
            test_y[np.squeeze(test_y) == 1],
            pred_test_y_bin[np.squeeze(test_y) == 1]
        )
        acc_test_neurotypical = get_acc_safe(
            test_y[np.squeeze(test_y) == 0],
            pred_test_y_bin[np.squeeze(test_y) == 0]
        )
        n = test.shape[0]
        n_asd = sum(test_y == 1)[0]
        n_neurotypical = sum(test_y == 0)[0]

        # bias_1 = true_positive_bias
        # bias_0 = false_positive_bias

        _, true_positive_bias, false_positive_bias = compute_error_and_bias(
            test_y.reshape(-1).astype(int),
            pred_test_y_bin.astype(int),
            (test_sex_data[:, 0]).astype(int),
            absolute_value=True)
        scores_dict["false_positive_bias"] = false_positive_bias
        scores_dict["true_positive_bias"] = true_positive_bias
        print('false_positive_bias', false_positive_bias)
        print('true_positive_bias', true_positive_bias)

    if population == 'male':
        auc_test = get_roc_auc_safe(
            test_y[test_sex_data[:, 0] == 1],
            pred_test_y[test_sex_data[:, 0] == 1][:, 1]
        )
        acc_test = get_acc_safe(
            test_y[test_sex_data[:, 0] == 1],
            pred_test_y_bin[test_sex_data[:, 0] == 1]
        )

        # Accuracy per DX_GROUP
        acc_test_asd = get_acc_safe(
            test_y[test_sex_data[:, 0] == 1][
                np.squeeze(test_y[test_sex_data[:, 0] == 1]) == 1],
            pred_test_y_bin[test_sex_data[:, 0] == 1][
                np.squeeze(test_y[test_sex_data[:, 0] == 1]) == 1]
        )
        acc_test_neurotypical = get_acc_safe(
            test_y[test_sex_data[:, 0] == 1][
                np.squeeze(test_y[test_sex_data[:, 0] == 1]) == 0],
            pred_test_y_bin[test_sex_data[:, 0] == 1][
                np.squeeze(test_y[test_sex_data[:, 0] == 1]) == 0]
        )
        n = sum(test_sex_data[:, 0] == 1)
        n_asd = sum(test_y[test_sex_data[:, 0] == 1] == 1)[0]
        n_neurotypical = sum(test_y[test_sex_data[:, 0] == 1] == 0)[0]

    if population == 'female':
        auc_test = get_roc_auc_safe(
            test_y[test_sex_data[:, 1] == 1],
            pred_test_y[test_sex_data[:, 1] == 1][:, 1]
        )
        acc_test = get_roc_auc_safe(
            test_y[test_sex_data[:, 1] == 1],
            pred_test_y_bin[test_sex_data[:, 1] == 1]
        )

        # Accuracy per DX_GROUP
        acc_test_asd = get_acc_safe(
            test_y[test_sex_data[:, 1] == 1][
                np.squeeze(test_y[test_sex_data[:, 1] == 1]) == 1],
            pred_test_y_bin[test_sex_data[:, 1] == 1][
                np.squeeze(test_y[test_sex_data[:, 1] == 1]) == 1]
        )
        acc_test_neurotypical = get_acc_safe(
            test_y[test_sex_data[:, 1] == 1][
                np.squeeze(test_y[test_sex_data[:, 1] == 1]) == 0],
            pred_test_y_bin[test_sex_data[:, 1] == 1][
                np.squeeze(test_y[test_sex_data[:, 1] == 1]) == 0]
        )
        n = sum(test_sex_data[:, 1] == 1)
        n_asd = sum(test_y[test_sex_data[:, 1] == 1] == 1)[0]
        n_neurotypical = sum(test_y[test_sex_data[:, 1] == 1] == 0)[0]

    print('AUC', auc_test)
    print('acc', acc_test)
    print('acc_asd', acc_test_asd)
    print('acc_neurotypical', acc_test_neurotypical)
    print('n', n)
    print('n_asd', n_asd)
    print('n_neurotypical', n_neurotypical)
    scores_dict.update(
        {
            "auc": auc_test,
            "acc": acc_test,
            "acc_asd": acc_test_asd,
            "acc_neurotypical": acc_test_neurotypical,
            "n": n,
            "n_asd": n_asd,
            "n_neurotypical": n_neurotypical,
        }
    )
    return scores_dict


def process_scores(scores, y, sex_data):
    metrics = ["auc", "acc", "acc_asd", "acc_neurotypical", "n", "n_asd",
               "n_neurotypical"]
    score_keys = [
                     (metric, sex) for metric in metrics for sex in
                     ["male", "female", "overall"]
                 ] + [("bias", "FP"), ("bias", "TP")]

    scores_dict = {key: [] for key in score_keys}

    for score_tuple in scores:
        pred = score_tuple[0]
        test_ind = score_tuple[-1]
        print('\nOverall')
        overall_scores = get_auc_acc(y, pred, test_ind, sex_data)
        for metric in metrics:
            scores_dict[(metric, "overall")].append(overall_scores[metric])
        scores_dict[("bias", "FP")].append(
            overall_scores["false_positive_bias"])
        scores_dict[("bias", "TP")].append(
            overall_scores["true_positive_bias"])

        print('\nMale')
        male_scores = get_auc_acc(y, pred, test_ind, sex_data, 'male')
        for metric in metrics:
            scores_dict[(metric, "male")].append(male_scores[metric])

        print('\nFemale')
        female_scores = get_auc_acc(y, pred, test_ind, sex_data, 'female')
        for metric in metrics:
            scores_dict[(metric, "female")].append(female_scores[metric])

    columns = pd.MultiIndex.from_tuples(score_keys,
                                        names=["metric", "category"])
    scores_df = pd.DataFrame(scores_dict, columns=columns)

    return scores_df


def run_cross_validation(strat_indices, graph, features, y, y_data, params,
                         subject_IDs, skf, num_nodes, sex_data=None,
                         stratify=False, save=False):
    flags_dict = tf.flags.FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        tf.flags.FLAGS.__delattr__(keys)
    tf.app.flags.DEFINE_string('f', '', 'kernel')

    if save == True:  # To save weights of a trained model
        fold_indices = range(10)
        with Pool(processes=10) as process_pool:
            scores = list(
                process_pool.starmap(
                    partial(
                        train_fold_thread,
                        graph_feat=graph,
                        features=features,
                        y=y,
                        y_data=y_data,
                        params=params,
                        subject_IDs=subject_IDs,
                        sex_data=sex_data,
                        stratify=stratify,
                    ),
                    zip(strat_indices, fold_indices)
                )
            )

    else:
        with Pool(processes=10) as process_pool:
            scores = list(
                process_pool.map(
                    partial(
                        train_fold_thread,
                        graph_feat=graph,
                        features=features,
                        y=y,
                        y_data=y_data,
                        params=params,
                        subject_IDs=subject_IDs,
                        sex_data=sex_data,
                        stratify=stratify,
                    ),
                    # [(train_ind, test_ind, test_ind) for train_ind, test_ind in reversed(list(skf.split(np.zeros(num_nodes), np.squeeze(y))))]
                    strat_indices
                )
            )

    # print(stratify)

    return scores
