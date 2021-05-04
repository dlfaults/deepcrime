import os
import glob
import csv
import numpy as np
import collections
from stats import power

import utils.properties as props
from run_deepcrime_properties import read_properties


def get_overall_mutation_score(stats_dir, train_stats_dir, accuracy_dir, lower_lr, prefix, train_accuracy_dir, killed_name_list):
    mutation_score = 0
    operator_num = 0
    excluded_num = 0
    score_dict = {}

    for filename in glob.glob(os.path.join(stats_dir, "*")):
        if '.csv' in filename:
            if is_binary_search_operator(filename):
                test_score, ins_score = get_binary_search_operator_mutation_score(filename, train_stats_dir, accuracy_dir, lower_lr, stats_dir, prefix, killed_name_list)
            else:
                test_score, ins_score = get_exhaustive_operator_mutation_score(filename, train_stats_dir, train_accuracy_dir, accuracy_dir, stats_dir, prefix)

            if test_score != -1:
                score_dict[filename.replace(stats_dir + os.path.sep, '')] = {'test_score': test_score,
                                                                             'ins_score': ins_score}

    return score_dict


def get_binary_search_operator_mutation_score(filename, train_stats_dir, accuracy_dir, lower_lr, stats_dir, prefix, killed_name_list):
    file_short_name = get_file_short_name(filename)
    train_killed_conf = get_killed_conf(os.path.join(train_stats_dir, file_short_name), train_stats_dir, killed_name_list)
    test_killed_conf = get_killed_conf(filename, train_stats_dir, killed_name_list)

    if train_killed_conf == -1:
        # mutant not killed by train set
        return -1, 10

    if test_killed_conf == -1:
        test_killed_conf = get_upper_bound(file_short_name, lower_lr)

    upper_bound = get_upper_bound(file_short_name, lower_lr)

    if train_killed_conf == test_killed_conf:
        mutation_score = 1
    elif upper_bound == train_killed_conf:
        mutation_score = -1
    else:
        mutation_score = round((upper_bound - test_killed_conf) / (upper_bound - train_killed_conf), 2)

    test_power_dict, ins_score_min, ins_score_max = get_power_dict_binary(accuracy_dir, filename, train_killed_conf,
                                                                          test_killed_conf, upper_bound, stats_dir, prefix)

    if mutation_score > 1:
        mutation_score = 1

    return mutation_score, abs(ins_score_max) + abs(ins_score_min)


def get_file_short_name(filename):
    return filename[filename.rindex(os.path.sep) + 1:len(filename)]


def get_upper_bound(file_short_name, lower_lr):
    if 'delete_td' in file_short_name:
        return 99

    if 'change_learning_rate' in file_short_name:
        return lower_lr

    if 'change_epochs' in file_short_name or 'change_patience' in file_short_name:
        return 1

    return 100


def get_power_dict_binary(accuracy_dir, stats_file_name, train_killed_conf, test_killed_conf, upper_bound, stats_dir, prefix):
    original_file = os.path.join(accuracy_dir, prefix + '.csv')
    original_accuracy = get_accuracy_array_from_file(original_file, 2)
    name = get_replacement_name(stats_file_name, stats_dir, prefix)
    overall_num = 0
    unstable_num = 0
    dict_for_binary = {}
    for filename in glob.glob(os.path.join(accuracy_dir, "*")):
        if name in filename:
            mutation_accuracy = get_accuracy_array_from_file(filename, 2)
            pow = power(original_accuracy, mutation_accuracy)

            mutation_parameter = filename.replace(accuracy_dir, '').replace('.csv', '').replace(name + '_', '').replace(
                'False_', '').replace('_0', '').replace('_3', '').replace('_9', '').replace('_1', '').replace(
                os.path.sep, '')

            if pow >= 0.8:
                dict_for_binary[float(mutation_parameter)] = 's'
            else:
                dict_for_binary[float(mutation_parameter)] = 'uns'

            dict_for_binary = collections.OrderedDict(sorted(dict_for_binary.items()))

    ins_score_min, ins_score_max = get_ins_score(stats_file_name, dict_for_binary, train_killed_conf, test_killed_conf,
                                                 upper_bound)
    return dict_for_binary, ins_score_min, ins_score_max


def get_power_dict_exh(accuracy_dir, stats_file_name, stats_dir, prefix):
    original_file = os.path.join(accuracy_dir, prefix + '.csv')
    original_accuracy = get_accuracy_array_from_file(original_file, 2)
    name = get_replacement_name(stats_file_name, stats_dir, prefix)

    dict_for_exh = {}
    for filename in glob.glob(os.path.join(accuracy_dir, "*")):
        if name in filename:
            mutation_accuracy = get_accuracy_array_from_file(filename, 2)
            pow = power(original_accuracy, mutation_accuracy)

            mutation_parameter = filename.replace(accuracy_dir, '').replace('.csv', '').replace(name + '_', '').replace(
                'False_', '').replace(os.path.sep, '')
            if pow >= 0.8:
                dict_for_exh[mutation_parameter] = 's'
            else:
                dict_for_exh[mutation_parameter] = 'uns'

    return dict_for_exh


def get_ins_score(stats_file_name, dict_for_binary, train_killed_conf, test_killed_conf, upper_bound):
    found_first_stable = False
    unstable = 0
    stable = 200
    for key in dict_for_binary:
        if dict_for_binary[key] == 'uns':
            unstable = float(key)
        elif dict_for_binary[key] == 's' and not found_first_stable and float(key) >= test_killed_conf:
            stable = float(key)
            found_first_stable = True

    if stable == 200 or (unstable > stable and not (
            'change_epochs' in stats_file_name or 'change_learning_rate' in stats_file_name or 'change_patience' in stats_file_name)):
        return 1, 1

    if stable < unstable and train_killed_conf < unstable and (
            'change_epochs' in stats_file_name or 'change_learning_rate' in stats_file_name or 'change_patience' in stats_file_name):
        return 0, 0

    if unstable < train_killed_conf and not (
            'change_epochs' in stats_file_name or 'change_learning_rate' in stats_file_name or 'change_patience' in stats_file_name):
        return 0, 0

    if upper_bound - train_killed_conf == 0 or unstable == 0:
        return 0, 0

    if unstable == upper_bound:
        return 1, 1

    if 'change_epochs' in stats_file_name or 'change_patience' in stats_file_name:
        upper_bound = 1

    ins_score_min = round(abs(unstable - train_killed_conf) / abs(upper_bound - train_killed_conf), 2)
    ins_score_max = round(abs(stable - train_killed_conf) / abs(upper_bound - train_killed_conf), 2)
    return ins_score_min, ins_score_max


def get_accuracy_array_from_file(filename, row_index):
    accuracy = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if any(x.strip() for x in row):
                accuracy.append(row[row_index])

    return np.asarray(accuracy).astype(np.float32)


def get_killed_conf(filename, train_stats_dir, killed_name_list):
    killed_conf = -1

    row_num = 0
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            killed_conf = row[0]

            row_num = row_num + 1

    if row_num == 1:
        return -1

    if killed_conf != -1 and train_stats_dir in filename:
        file_short_name = filename[filename.rindex(os.path.sep) + 1:len(filename)]
        killed_name_list.append(file_short_name + '_' + killed_conf)

    return float(killed_conf)


def get_exhaustive_operator_mutation_score(filename, train_stats_dir, train_accuracy_dir, accuracy_dir, stats_dir, prefix):
    power_dict_exh_train = get_power_dict_exh(train_accuracy_dir, filename, stats_dir, prefix)
    power_dict_exh_test = get_power_dict_exh(accuracy_dir, filename, stats_dir, prefix)

    file_short_name = filename[filename.rindex(os.path.sep) + 1:len(filename)]
    train_killed_conf = get_killed_from_csv(os.path.join(train_stats_dir, file_short_name), train_stats_dir)

    if len(train_killed_conf) == 0:
        return -1, -1

    test_killed_conf = get_killed_from_csv(filename, train_stats_dir)

    for killed_conf in train_killed_conf:
        if power_dict_exh_train.get(killed_conf) == 'uns':
            train_killed_conf.remove(killed_conf)

    killed_conf = np.intersect1d(train_killed_conf, test_killed_conf)

    if len(train_killed_conf) == 0:
        mutation_score = 0
    else:
        mutation_score = round(len(killed_conf) / len(train_killed_conf), 2)

    if not len(killed_conf) == 0:
        ins_score = get_ins_score_exh(killed_conf, power_dict_exh_test)
    else:
        ins_score = 0

    return mutation_score, ins_score


def get_ins_score_exh(killed_conf, power_dict_exh_test):
    ins_num = 0
    for kc in killed_conf:
        if power_dict_exh_test.get(kc) == 'uns':
            ins_num = ins_num + 1

    return round(ins_num / len(killed_conf), 2)


def get_killed_from_csv(filename, train_stats_dir):
    index = get_outcome_row_index(filename)

    killed_conf = []
    killed_count = 0
    row_count = 0
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[index] == 'TRUE' or row[index] == 'True':
                if train_stats_dir in filename:
                    file_short_name = filename[filename.rindex(os.path.sep) + 1:len(filename)]

                killed_count = killed_count + 1

                if 'l1' in row[0] or 'l2' in row[0] or 'l1_l2' in row[0]:
                    param = row[0][0: len(row[0]) - 2]
                else:
                    param = row[0]

                killed_conf.append(param)
            row_count = row_count + 1

    ratio = round(killed_count / row_count, 2)
    return killed_conf


def get_outcome_row_index(filename):
    if ('disable_batching' in filename) or ('remove_validation_set' in filename):
        return 2
    else:
        return 3


def is_binary_search_operator(filename):
    operator_list = ['change_label', 'delete_td', 'unbalance_td', 'add_noise',
                     'output_classes_overlap', 'change_epochs', 'change_learning_rate', 'change_patience']
    for operator in operator_list:
        if operator in str(filename):
            return True

    return False


def get_replacement_name(stats_file_name, stats_dir, prefix):
    killed_mutation = stats_file_name.replace(stats_dir + os.path.sep, prefix + '_')
    killed_mutation = killed_mutation.replace('_exssearch.csv', '_mutated0_MP')
    if 'change_epochs' in killed_mutation and 'udacity' in stats_dir:
        killed_mutation = killed_mutation.replace('_binarysearch.csv', '_mutated0_MP_50')
    if 'change_learning_rate' in killed_mutation or 'change_epochs' in killed_mutation:
        killed_mutation = killed_mutation.replace('_binarysearch.csv', '_mutated0_MP_False')
    else:
        killed_mutation = killed_mutation.replace('_binarysearch.csv', '_mutated0_MP')

    killed_mutation = killed_mutation.replace('_nosearch.csv', '_mutated0_MP')
    killed_mutation = killed_mutation.replace('unbalance_td', 'unbalance_train_data')
    killed_mutation = killed_mutation.replace('delete_td', 'delete_training_data')
    killed_mutation = killed_mutation.replace('output_classes_overlap', 'make_output_classes_overlap')
    killed_mutation = killed_mutation.replace('change_patience', 'change_earlystopping_patience')
    return killed_mutation


def get_mutation_score(score_dict):
    overall_mutation_score = 0
    operator_num = 0
    for key in score_dict:
        if score_dict[key]['ins_score'] == 0:
            overall_mutation_score = overall_mutation_score + score_dict[key]['test_score']
            operator_num = operator_num + 1

    if operator_num == 0:
        overall_mutation_score = 0
    else:
        overall_mutation_score = overall_mutation_score / operator_num

    return overall_mutation_score


# if __name__ == "__main__":
def calculate_dc_ms(train_accuracy_dir, accuracy_dir):
    dc_props = read_properties()
    subject_name = dc_props['subject_name']
    prefix = subject_name

    model_params = getattr(props, "model_properties")

    epochs = model_params["epochs"]

    lr_params = getattr(props, "change_learning_rate")

    lower_lr = lr_params["bs_upper_bound"]
    upper_lr = lr_params["bs_lower_bound"]

    killed_name_list = []

    train_stats_dir = os.path.join(train_accuracy_dir, 'stats')

    stats_dir = os.path.join(accuracy_dir, 'stats')
    score_dict = get_overall_mutation_score(stats_dir, train_stats_dir, accuracy_dir, lower_lr, prefix, train_accuracy_dir, killed_name_list)

    mut_score = get_mutation_score(score_dict)


    ms_csv_file = os.path.join(accuracy_dir, subject_name + "_ms.csv")

    with open(ms_csv_file, 'w') as f1:
        writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
        writer.writerow(['operator_name', 'operator_ms', 'operator_instability_score'])
        for key, value in score_dict.items():
            writer.writerow([key, value['test_score'], value['ins_score']])
        writer.writerow(['total MS:', mut_score, ''])

