import numpy as np
import matplotlib.pyplot as plt

# Load data
X_train = np.loadtxt('svm_training_inputs.txt')
y_train = np.loadtxt('svm_training_targets.txt')
X_test = np.loadtxt('svm_test_inputs.txt')

# Plot training data
plt.figure(figsize=(8, 6))
for label in np.unique(y_train):
    plt.scatter(X_train[y_train == label, 0], X_train[y_train == label, 1], label=f'Class {int(label)}')

# Plot test data
plt.scatter(X_test[:, 0], X_test[:, 1], color='black', marker='x', label='Test Points')

# Add decision boundary (your guess, adjust as needed)
x = np.linspace(-2, 12, 100)
y = 0.5 * x + 2  # Example guessed boundary
plt.plot(x, y, color='red', label='Guessed Decision Boundary')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Training and Test Data with Guessed Decision Boundary')
plt.legend()
plt.grid()
plt.show()
