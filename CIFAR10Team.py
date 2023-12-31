import numpy as np
from keras.datasets import cifar10
import ssl
import importlib  
from matplotlib import pyplot as plt
import os

ssl._create_default_https_context = ssl._create_unverified_context

(X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()

Y_train=Y_train.reshape(Y_train.shape[0])
Y_test=Y_test.reshape(Y_test.shape[0])

def loadtxt(filename):
    if os.path.isfile(filename):
        tmp = np.loadtxt(filename, delimiter=',')
        return tmp/(np.max(tmp) - np.min(tmp))
    else:
        return []
 

Y_test_scores_threshold = loadtxt("class_sums/CIFAR10AdaptiveThresholding_99_2000_500_10.0_10_32_1.txt")
Y_test_scores_thermometer_3 = loadtxt("class_sums/CIFAR10ColorThermometers_99_2000_1500_2.5_3_8_32_1.txt")
Y_test_scores_thermometer_4 = loadtxt("class_sums/CIFAR10ColorThermometers_99_2000_1500_2.5_4_8_32_1.txt")
Y_test_scores_hog = loadtxt("class_sums/CIFAR10HistogramOfGradients_99_2000_50_10.0_0_32_0.txt")
sauvola = loadtxt("class_sums/CIFAR10SauvolaThreshold_100_epochs.txt")
niblack = loadtxt("class_sums/CIFAR10NiblackThreshold_10_epochs.txt")
canny = loadtxt("class_sums/Canny_Edge_Detection_100_epochs.txt")

def vote_factor(probabilities, i):
    i_max = np.max(probabilities[i])
    if i_max < 0:
        return probabilities[i]*0.9
    ''' if i_max < 0.003:
        return probabilities[i]/2'''
    max = np.max(probabilities) 
    return probabilities[i]#*(2+3*i_max/max)
    probabilities -= np.min(probabilities) - 1
    return 1#(np.sum(probabilities * np.log2(probabilities)))



votes = np.zeros(canny.shape, dtype=np.float32)
for i in range(Y_test.shape[0]):
    votes[i] += 1.3*vote_factor(Y_test_scores_threshold, i)
    votes[i] += 2*vote_factor(Y_test_scores_thermometer_3, i)
    votes[i] += 2.4*vote_factor(Y_test_scores_thermometer_4, i)
    votes[i] += 2.5*vote_factor(Y_test_scores_hog, i)
    votes[i] += 0.7*vote_factor(sauvola, i)
    votes[i] += 0.7*vote_factor(niblack, i)
    votes[i] += 0.5*vote_factor(canny, i)

Y_test_predicted = votes.argmax(axis=1)

print("Team Accuracy: %.1f" % (100*(Y_test_predicted == Y_test).mean()))
wrong = []
for i, img_class in enumerate(Y_test):
    if img_class !=  Y_test_predicted[i]:
        wrong.append(img_class)

print(len(wrong))

labels = ["plane", "cat"]
for i, label in enumerate(labels):
    plt.text(i, label, "")
plt.hist(wrong)

plt.show()

''' Y_test_scores_threshold = Y_test_scores_threshold/(np.max(Y_test_scores_threshold) - np.min(Y_test_scores_threshold))
Y_test_scores_thermometer_3 = Y_test_scores_thermometer_3/(np.max(Y_test_scores_thermometer_3) - np.min(Y_test_scores_thermometer_3))
Y_test_scores_thermometer_4 = Y_test_scores_thermometer_4/(np.max(Y_test_scores_thermometer_4) - np.min(Y_test_scores_thermometer_4))
Y_test_scores_hog = Y_test_scores_hog/(np.max(Y_test_scores_hog) - np.min(Y_test_scores_hog))

def vote_factor(probabilities):
    print(probabilities)
    probabilities -= np.min(probabilities) - 1
    return -np.sum(probabilities * np.log2(probabilities))



votes = np.zeros(Y_test_scores_threshold.shape, dtype=np.float32)
for i in range(Y_test.shape[0]):
    print("Class:", Y_test[i])
    votes[i] += vote_factor(Y_test_scores_threshold[i])*Y_test_scores_threshold[i]
    print(vote_factor(Y_test_scores_threshold[i]))
    votes[i] += vote_factor(Y_test_scores_thermometer_3[i])*Y_test_scores_thermometer_3[i]
    print(vote_factor(Y_test_scores_thermometer_3[i]))
    votes[i] += vote_factor(Y_test_scores_thermometer_4[i])*Y_test_scores_thermometer_4[i]
    print(vote_factor(Y_test_scores_thermometer_4[i]))
    votes[i] += vote_factor(Y_test_scores_hog[i])*Y_test_scores_hog[i]
    print(vote_factor(Y_test_scores_hog[i]))
    if(i == 2):
        exit()

Y_test_predicted = votes.argmax(axis=1)

print("Team Accuracy: %.1f" % (100*(Y_test_predicted == Y_test).mean()))
print(Y_test.shape)
 '''