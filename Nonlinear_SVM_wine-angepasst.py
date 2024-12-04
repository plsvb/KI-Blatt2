"""
================================================
SVM: Maximum margin separating decision boundary
================================================

Apply nonlinear SVM to wine data set.
"""
print(__doc__)

import numpy as np
from sklearn import svm

# Load the dataset
X = np.loadtxt('wine_inputs.txt')  # Features
tmp = np.loadtxt('wine_targets.txt')  # Targets
Y = np.array([[tmp[i]] for i in range(tmp.size)])
Y.shape = tmp.size

# Split the data into training and test sets
# Training: 2/3 of the samples, Test: 1/3 of the samples
numTestSamples = int(tmp.size / 3 + 1)
numTrainingSamples = tmp.size - numTestSamples

training = np.zeros((numTrainingSamples, X.shape[1]))
target_training = np.zeros(numTrainingSamples)
test = np.zeros((numTestSamples, X.shape[1]))
target_test = np.zeros(numTestSamples)

index_train = 0
index_test = 0

for i in range(tmp.size):
    if np.mod(i, 3) == 0:  # Assign every third sample to the test set
        test[index_test] = X[i]
        target_test[index_test] = Y[i]
        index_test += 1
    else:  # Assign the rest to the training set
        training[index_train] = X[i]
        target_training[index_train] = Y[i]
        index_train += 1

print(f"Input dimension: {X.shape[1]}")
print(f"Number of training samples: {numTrainingSamples}")
print(f"Number of test samples: {numTestSamples}\n")

# Define kernels and their parameters to test
kernels = [
    {'kernel': 'rbf', 'C': 1, 'gamma': 0.1},
    {'kernel': 'poly', 'C': 1, 'degree': 3, 'gamma': 'scale'},
]

# Loop over different kernel settings
for kernel_params in kernels:
    print(f"Training SVM with kernel: {kernel_params['kernel']}, parameters: {kernel_params}")
    clf = svm.SVC(**kernel_params)  # Initialize the classifier with the kernel settings
    clf.fit(training, target_training)  # Train the model

    # Calculate training error
    training_error_svm = 0
    output_training = clf.predict(training)
    for i in range(numTrainingSamples):
        if target_training[i] != output_training[i]:
            training_error_svm += 1
    training_error_svm = training_error_svm / numTrainingSamples
    print(f"Number of misclassified training patterns: {int(training_error_svm * numTrainingSamples)}")
    print(f"Training error SVM: {training_error_svm:.4f}")

    # Calculate test error
    test_error_svm = 0
    output_test = clf.predict(test)
    for i in range(numTestSamples):
        if target_test[i] != output_test[i]:
            test_error_svm += 1
    test_error_svm = test_error_svm / numTestSamples
    print(f"Number of misclassified test patterns: {int(test_error_svm * numTestSamples)}")
    print(f"Test error SVM: {test_error_svm:.4f}\n")
