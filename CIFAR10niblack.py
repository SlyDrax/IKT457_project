import argparse
from tmu.models.classification.vanilla_classifier import TMClassifier
import numpy as np
from keras.datasets import cifar10
import cv2
from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer
from time import time
from skimage.color import rgb2gray
from skimage.color import gray2rgb
import matplotlib.pyplot as plt
import skimage as ski
import scipy as sp
from skimage import feature

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clauses", default=2000, type=int)
    parser.add_argument("--T", default=500, type=int)
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--max_included_literals", default=32, type=int)
    parser.add_argument("--device", default="GPU", type=str)
    parser.add_argument("--weighted_clauses", default=True, type=bool)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--patch_size", default=10, type=int)

    args = parser.parse_args()

    (X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()
    
    X_train = np.copy(X_train_org)
    X_test = np.copy(X_test_org)
    
    Y_train=Y_train.reshape(Y_train.shape[0])
    Y_test=Y_test.reshape(Y_test.shape[0])
    window_size = 25
    for i in range(X_train.shape[0]):
        niblack = ski.filters.threshold_niblack(X_train_org[i], window_size=window_size, k=0.8)
        binary_niblack = X_train_org[i] > niblack
        X_train[i] = binary_niblack


    for i in range(X_test.shape[0]):
        niblack = ski.filters.threshold_niblack(X_test_org[i], window_size=window_size, k=0.8)
        binary_niblack = X_test_org[i] > niblack
        X_test[i] = binary_niblack

    tm = TMClassifier(
        number_of_clauses=args.num_clauses,
        T=args.T,
        s=args.s,
        max_included_literals=args.max_included_literals,
        platform=args.device,
        weighted_clauses=args.weighted_clauses,
        patch_dim=(args.patch_size, args.patch_size)
    ) 

    mini_batch_size = 2000
    result_test = None
    for epoch in range(args.epochs):
        start_training = time()
        for i in range(0, len(X_train), mini_batch_size):
            X_mini_batch = X_train[i:i+mini_batch_size]
            Y_mini_batch = Y_train[i:i+mini_batch_size]
            tm.fit(X_mini_batch, Y_mini_batch)
        stop_training = time()

        if (epoch + 1) % 10 == 0:
            start_testing = time()
            Y_test_predicted, Y_test_scores = tm.predict(X_test, return_class_sums=True)
            stop_testing = time()

            result_test = 100*(Y_test_scores.argmax(axis=1) == Y_test).mean()

            print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (epoch+1, result_test, stop_training-start_training, stop_testing-start_testing))

            np.savetxt("training/CIFAR10NiblackThreshold%d_%d_%d_%.1f_%d_%d_%d.txt" % (epoch+1, args.num_clauses, args.T, args.s, args.patch_size, args.max_included_literals, args.weighted_clauses), Y_test_scores, delimiter=',')

    Y_test_predicted, Y_test_scores = tm.predict(X_test, return_class_sums=True)
    result_test = 100*(Y_test_scores.argmax(axis=1) == Y_test).mean()

    np.savetxt("class_sums/CIFAR10NiblackThreshold_%d_ephocs.txt" % (args.epochs), Y_test_scores, delimiter=',') 

    print("#%d Accuracy: %.2f%%" % (args.epochs, result_test))




