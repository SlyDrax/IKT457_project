import argparse
from tmu.models.classification.vanilla_classifier import TMClassifier
import numpy as np
from keras.datasets import cifar10
import cv2
from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer
from time import time

patch_size = 0

blockSize = 2
ksize = 3
k = 0.08
maxCorners = 30
qualityLevel = 0.05
minDistance = 10

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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

    for i in range(X_train.shape[0]):
        gray = cv2.cvtColor(X_train_org[i],cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        corners = cv2.cornerHarris(gray, blockSize, ksize, k)
        dst = cv2.dilate(corners,None)
        X_train[i][dst>qualityLevel*dst.max()]=[255,255,255]
        X_train[i][dst<=qualityLevel*dst.max()]=[0,0,0]

    for i in range(X_test.shape[0]):
        gray = cv2.cvtColor(X_test_org[i],cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        corners = cv2.cornerHarris(gray, blockSize, ksize, k)
        dst = cv2.dilate(corners,None)
        X_test[i][dst>qualityLevel*dst.max()]=[255,255,255]
        X_test[i][dst<=qualityLevel*dst.max()]=[0,0,0]

    tm = TMClassifier(
        number_of_clauses=args.num_clauses,
        T=args.T,
        s=args.s,
        max_included_literals=args.max_included_literals,
        platform=args.device,
        weighted_clauses=args.weighted_clauses,
        patch_dim=(args.patch_size, args.patch_size)
    ) 
    print(X_train.shape)
    print(Y_train.shape)
    for epoch in range(args.epochs):
        start_training = time()
        tm.fit(X_train, Y_train)
        stop_training = time()

        start_testing = time()
        Y_test_predicted, Y_test_scores = tm.predict(X_test, return_class_sums=True)
        stop_testing = time()

        result_test = 100*(Y_test_scores.argmax(axis=1) == Y_test).mean()

        print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (epoch+1, result_test, stop_training-start_training, stop_testing-start_testing))
        
        np.savetxt("training/CIFAR10CornerDetection_%d_%d_%d_%.2f_%d_%d_%d.txt" % (epoch+1, args.num_clauses, args.T, args.s, patch_size, args.max_included_literals, args.weighted_clauses), Y_test_scores, delimiter=',') 

    np.savetxt("class_sums/CIFAR10CornerDetection_%d_ephocs.txt" % (args.epochs), Y_test_predicted, delimiter=',') 



