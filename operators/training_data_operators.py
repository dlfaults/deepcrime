# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 13:36:29 2020

@author: anonymous
"""
from os import uname_result

import numpy as np
import random
from utils import properties


def operator_change_labels(y_train, label=None, percentage=-1):
    if properties.model_type == 'regression':
        y_train_noisy = operator_add_noise_to_labels(y_train, percentage)
        return y_train_noisy
    else:
        # get the unique elements of the array with index and count information
        unique_label_list, unique_inverse, unique_counts = np.unique(y_train, return_counts=True, return_inverse=True,
                                                                     axis=0)
        if label is None:
            # index of the most often occurring label
            index = np.argmax(unique_counts)
            # label with the highest number of occurrence
            label = unique_label_list[index]
            print("Selected Label is:" + str(label))

        # all the indices of the selected label in the original array
        indexes_of_label = np.argwhere(unique_inverse == index)

        y_train_replaced = mutate_labels(y_train, label, unique_label_list, indexes_of_label, index, percentage)
        return y_train_replaced


def operator_add_noise_to_labels(y_train, percentage=-1):
    sigma = np.std(y_train)
    sigma_percentage = abs(sigma * percentage / 100)
    noise = np.random.normal(0, sigma_percentage, y_train.shape)
    noisy_data = y_train + noise
    return noisy_data


def mutate_labels(y_train, label, label_list, args, index, percentage):
    if percentage == 100:
        indexes_to_replace = args
    else:
        indexes_to_replace = get_random_indexes(args, percentage)

    y_train_randomly_replaced = np.copy(y_train)
    # replacement_values are all the other labels available
    replacement_values = np.delete(label_list, index, axis=0)

    for j in range(len(indexes_to_replace)):
        replacement_value = random.choice(replacement_values)
        y_train_randomly_replaced[indexes_to_replace[j]] = replacement_value

    assert (len(y_train) == len(y_train_randomly_replaced))
    return y_train_randomly_replaced


# deletes portion of training data in a balanced way
def operator_delete_training_data(x_train, y_train, percentage=-1):
    return delete_training_data(x_train, y_train, percentage, 'DELETE')


def unbalance_training_data(x_train, y_train, percentage=-1):
    return delete_training_data(x_train, y_train, percentage, 'UNBALANCE')


def operator_make_output_classes_overlap(x_train, y_train, percentage=-1, label1=None, label2=None):
    x_train_is_list = isinstance(x_train, list)

    if x_train_is_list:
        x_train_overlapped = []
        for i in range(len(x_train)):
            x_train_overlapped.append(np.copy(x_train[i]))
    else:
        x_train_overlapped = np.copy(x_train)

    y_train_overlapped = np.copy(y_train)
    if properties.model_type == 'regression':
        unique_label_list, unique_counts, unique_inverse = get_label_buckets(y_train)
    else:
        unique_label_list, unique_inverse, unique_counts = np.unique(y_train_overlapped, return_counts=True,
                                                                     return_inverse=True, axis=0)

    print(unique_label_list)

    if label1 is None:
        index1 = np.argmax(unique_counts)
        label1 = unique_label_list[index1]
        unique_counts[index1] = -1

    if label2 is None:
        index2 = np.argmax(unique_counts)
        label2 = unique_label_list[index2]

    print("Label 1 is:" + str(label1))
    print("Label 2 is:" + str(label2))

    indexes_of_label1 = np.argwhere(unique_inverse == index1)

    indexes_to_duplicate = get_random_indexes(indexes_of_label1, percentage).squeeze()

    if x_train_is_list:
        for i in range(len(x_train)):
            x_train_portion = np.take(x_train_overlapped[i], indexes_to_duplicate, axis=0)
            x_train_overlapped[i] = np.concatenate((x_train_overlapped[i], x_train_portion))
    else:
        x_train_portion = np.take(x_train_overlapped, indexes_to_duplicate, axis=0)
        x_train_overlapped = np.concatenate((x_train_overlapped, x_train_portion))

    if properties.model_type == 'regression':
        indexes_of_label2 = np.argwhere(unique_inverse == index2)
        y_train_portion_label2 = np.take(y_train_overlapped, indexes_of_label2, axis=0)
        unique_label_list, unique_inverse, unique_counts = np.unique(y_train_portion_label2, return_counts=True,
                                                                     return_inverse=True, axis=0)

        y_train_portion = np.empty(y_train.shape, dtype=np.int8)
        unique_label_list = unique_label_list.squeeze()
        for i in range(0, round(len(indexes_to_duplicate) / len(unique_label_list) + 0.5)):
            y_train_overlapped = np.concatenate((y_train_overlapped, unique_label_list))

        diff = abs(len(y_train_overlapped) - (len(y_train) + len(indexes_to_duplicate)))
        y_train_overlapped = y_train_overlapped[0:len(y_train_overlapped) - diff]
    else:
        y_train_portion = np.take(y_train_overlapped, indexes_to_duplicate, axis=0)
        y_train_overlapped = np.concatenate((y_train_overlapped, np.full(y_train_portion.shape, label2)))
    return x_train_overlapped, y_train_overlapped


def delete_training_data(x_train, y_train, percentage, operator):
    x_train_is_list = isinstance(x_train, list)

    if properties.model_type == 'regression':
        unique_label_list, unique_counts, unique_inverse = get_label_buckets(y_train)
    else:
        unique_label_list, unique_counts = np.unique(y_train, return_counts=True, axis=0)

    average = np.mean(unique_counts)

    if x_train_is_list:
        x_train_to_delete = []
        for i in range(0, len(x_train)):
            x_train_to_delete.append(np.copy(x_train[i]))
    else:
        x_train_to_delete = np.copy(x_train)

    y_train_to_delete = np.copy(y_train)

    index = 0
    args_to_delete = []
    for label in unique_label_list:
        if properties.model_type == 'regression':
            args = np.argwhere(unique_inverse == index)
            if (operator == 'DELETE') or (operator == 'UNBALANCE' and len(args) < average):
                indexes_to_delete = get_random_indexes(args, percentage)
                args_to_delete.extend(indexes_to_delete.flatten().tolist())
        else:
            unique_label_list_r, unique_inverse, unique_counts = np.unique(y_train_to_delete, return_counts=True,
                                                                           return_inverse=True,
                                                                           axis=0)

            if (operator == 'DELETE') or (operator == 'UNBALANCE' and len(args) < average):
                args = np.argwhere(unique_inverse == index)
                indexes_to_delete = get_random_indexes(args, percentage)
                print("deleting " + str(len(indexes_to_delete)) + "  from " + str(len(args)))
                y_train_to_delete = np.delete(y_train_to_delete, indexes_to_delete, axis=0)

                if x_train_is_list:
                    for i in range(0, len(x_train_to_delete)):
                        x_train_to_delete[i] = np.delete(x_train_to_delete[i], indexes_to_delete, axis=0)
                else:
                    x_train_to_delete = np.delete(x_train_to_delete, indexes_to_delete, axis=0)
        index = index + 1

    if properties.model_type == 'regression':
        y_train_to_delete = np.delete(y_train_to_delete, args_to_delete, axis=0)
        if x_train_is_list:
            for i in range(0, len(x_train_to_delete)):
                x_train_to_delete[i] = np.delete(x_train_to_delete[i], args_to_delete, axis=0)
        else:
            x_train_to_delete = np.delete(x_train_to_delete, args_to_delete, axis=0)

    return (x_train_to_delete, y_train_to_delete)


def get_label_buckets(y_train):
    shape = y_train.shape
    if len(shape) == 1:
        return get_label_buckets_1dim(y_train)
    else:
        return get_label_buckets_2dims(y_train)


def get_label_buckets_1dim(y_train):
    std_array = np.std(y_train)

    index = 0
    bucket1 = []
    bucket2 = []
    bucket3 = []

    for element in y_train:
        if element < -std_array:
            bucket1.append(element)
        elif -std_array <= element <= std_array:
            bucket2.append(element)
        elif element > std_array:
            bucket3.append(element)

    buckets = [bucket1, bucket2, bucket3]

    unique_label_list = (0, 1, 2)
    unique_inverse = [-1] * len(y_train)
    unique_count = [-1] * len(unique_label_list)
    bucket_ind = 0
    for bucket in buckets:
        unique_count[bucket_ind] = len(bucket)
        for element in bucket:
            indices = np.argwhere(y_train == element)
            for index in indices:
                unique_inverse[index[0]] = bucket_ind

        bucket_ind = bucket_ind + 1
    return unique_label_list, np.asarray(unique_count), np.asarray(unique_inverse)


def get_label_buckets_2dims(y_train):
    std_array = np.std(y_train, axis=0)

    unique_label_list, unique_inverse, unique_counts = np.unique(y_train, return_counts=True,
                                                                 return_inverse=True,
                                                                 axis=0)
    index = 0
    bucket1 = []
    bucket2 = []
    bucket3 = []
    bucket4 = []
    bucket5 = []
    bucket6 = []
    bucket7 = []
    bucket8 = []
    bucket9 = []
    for label in unique_label_list:
        args = np.argwhere(unique_inverse == index)
        if label[0] < -std_array[0]:
            if label[1] < -std_array[1]:
                bucket1.extend(args.flatten().tolist())
            elif label[1] >= -std_array[1] and label[1] < std_array[1]:
                bucket2.extend(args.flatten().tolist())
            elif label[1] >= std_array[1]:
                bucket3.extend(args.flatten().tolist())
        elif -std_array[0] <= label[0] < std_array[0]:
            if label[1] < -std_array[1]:
                bucket4.extend(args.flatten().tolist())
            elif -std_array[1] <= label[1] < std_array[1]:
                bucket5.extend(args.flatten().tolist())
            elif label[1] >= std_array[1]:
                bucket6.extend(args.flatten().tolist())
        elif label[0] >= std_array[0]:
            if label[1] < -std_array[1]:
                bucket7.extend(args.flatten().tolist())
            elif -std_array[1] <= label[1] < std_array[1]:
                bucket8.extend(args.flatten().tolist())
            elif label[1] >= std_array[1]:
                bucket9.extend(args.flatten().tolist())
        index = index + 1

    buckets = [bucket1, bucket2, bucket3, bucket4, bucket5, bucket6, bucket7, bucket8, bucket9]

    unique_label_list = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    unique_inverse = [-1] * len(y_train)
    unique_count = [-1] * len(unique_label_list)
    bucket_ind = 0
    for bucket in buckets:
        unique_count[bucket_ind] = len(bucket)
        for element in bucket:
            unique_inverse[element] = bucket_ind

        bucket_ind = bucket_ind + 1
    return unique_label_list, np.asarray(unique_count), np.asarray(unique_inverse)


def operator_add_noise_to_training_data(x_train, percentage=-1):
    x_train_is_list = isinstance(x_train, list)

    if x_train_is_list:
        x_train_noisy = []
        for i in range(0, len(x_train)):
            x_train_noisy.append(np.copy(x_train[i]))
        indexes = np.arange(len(x_train[0]))
    else:
        x_train_noisy = np.copy(x_train)
        indexes = np.arange(len(x_train))

    if percentage == 100:
        indexes_to_replace = indexes
    else:
        indexes_to_replace = get_random_indexes(indexes, percentage)

    for j in indexes_to_replace:
        if x_train_is_list:
            image = x_train[0][j]
        else:
            image = x_train[j]

        dimension = image.shape
        reshaped = image.flatten()
        sigma = np.std(reshaped)
        sigma_percentage = sigma * percentage / 100

        noise = np.random.normal(0, sigma_percentage, dimension)
        noisy_data = image + noise
        if x_train_is_list:
            x_train_noisy[0][j] = noisy_data
        else:
            x_train_noisy[j] = noisy_data

    return x_train_noisy


def get_random_indexes(args, percentage):
    args_shuffled = args[np.random.permutation(len(args))]
    args_divided = np.array_split(args_shuffled, [0, int((len(args) * percentage) / 100)])
    random_indexes = args_divided[1]
    return random_indexes
