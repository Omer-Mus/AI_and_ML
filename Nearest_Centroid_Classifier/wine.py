#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Omer Mustel UNI:om2349
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sys, csv


def euclidean_distance(a, b):
    diff = a - b
    return np.sqrt(np.dot(diff, diff))


def load_data(csv_filename):
    """ 
    Returns a numpy ndarray in which each row repersents
    a wine and each column represents a measurement. There should be 11
    columns (the "quality" column cannot be used for classificaiton).
    """
    df = np.genfromtxt(csv_filename, delimiter=';', skip_header=1)
    return df[:, 0: df.shape[1] - 1]


def split_data(dataset, ratio=0.9):
    """
    Return a (train, test) tuple of numpy ndarrays. 
    The ratio parameter determines how much of the data should be used for 
    training. For example, 0.9 means that the training portion should contain
    90% of the data. You do not have to randomize the rows. Make sure that 
    there is no overlap. 
    """
    get_index = ratio * len(dataset)
    test_set, train_set = dataset[:int(get_index), :], dataset[int(get_index):, :]

    return test_set, train_set


def compute_centroid(data):
    """
    Returns a 1D array (a vector), representing the centroid of the data
    set. 
    """
    return sum(data)/len(data)


def experiment(ww_train, rw_train, ww_test, rw_test):
    """
    Train a model on the training data by creating a centroid for each class.
    Then test the model on the test data. Prints the number of total 
    predictions and correct predictions. Returns the accuracy. 
    """
    np.set_printoptions(suppress=True)
    white_centroid = compute_centroid(ww_train)
    red_centroid = compute_centroid(rw_train)
    # print("In experiment: ", red_centroid, white_centroid)

    correct_pred, wrong_pred = 0, 0

    for i in ww_test:
        dist_white = euclidean_distance(i, white_centroid)
        dist_red = euclidean_distance(i, red_centroid)
        if dist_white <= dist_red:
            correct_pred += 1
        else:
            wrong_pred += 1

    for j in rw_test:
        dist_white = euclidean_distance(j, white_centroid)
        dist_red = euclidean_distance(j, red_centroid)
        if dist_red <= dist_white:
            correct_pred += 1
        else:
            wrong_pred += 1

    total_predictions = len(ww_test) + len(rw_test)

    accuracy = correct_pred / total_predictions

    print("total number of predictions: ", total_predictions)
    print("number of correct predictions: ", correct_pred)
    print("Accuracy of the model: ", accuracy, "\n")

    return accuracy


def cross_validation(ww_data, rw_data, k):
    """
    Perform k-fold crossvalidation on the data and print the accuracy for each
    fold. 
    """
    k_acc = []
    partition = len(rw_data) // k

    for i in range(0, k):

        first = int(i * partition)
        last = int(first + np.ceil(partition))

        new_ww_test = ww_data[first: last, :]
        new_ww_train = np.concatenate((ww_data[:first, :], ww_data[last:, :]))
        # print("new_ww_test: ", new_ww_test.shape[0])
        new_rw_test = rw_data[first: last, :]
        new_rw_train = np.concatenate((rw_data[:first, :], rw_data[last:, :]))

        acc = experiment(new_ww_train, new_rw_train, new_ww_test, new_rw_test)
        k_acc.append(acc)

    accuracy = compute_centroid(k_acc)

    return accuracy

    
if __name__ == "__main__":
    
    ww_data = load_data('whitewine.csv')
    rw_data = load_data('redwine.csv')
    # Uncomment the following lines for step 2:
    ww_train, ww_test = split_data(ww_data, 0.9)
    rw_train, rw_test = split_data(rw_data, 0.9)
    experiment(ww_train, rw_train, ww_test, rw_test)
    
    # Uncomment the following lines for step 3:
    k = 10
    acc = cross_validation(ww_data, rw_data, k)
    print("{}-fold cross-validation accuracy: {}".format(k, acc))
    
