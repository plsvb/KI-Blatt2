# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Training data
X_train = np.array([[0, 0], [0, 2], [2, -1], [2, 3]])
y_train = np.array([-1, 1, -1, 1])

# Define test points
X_test = np.array([[0.5, 0.5], [1.5, 1.25]])

# Train the linear SVM
svm = SVC(kernel='linear', C=1e10)  # High C for hard margin SVM
svm.fit(X_train, y_train)

# Get the separating hyperplane
w = svm.coef_[0]
b = svm.intercept_[0]
x = np.linspace(-1, 3, 100)
y = -(w[0] / w[1]) * x - b / w[1]

# Get the margin boundaries
margin = 1 / np.linalg.norm(w)
y_margin_up = y + margin
y_margin_down = y - margin

# Plotting
plt.figure(figsize=(8, 6))
# Plot training data
for i, label in enumerate(y_train):
    if label == 1:
        plt.scatter(X_train[i, 0], X_train[i, 1], color='blue', label='Class 1' if i == 1 else "")
    else:
        plt.scatter(X_train[i, 0], X_train[i, 1], color='red', label='Class -1' if i == 0 else "")

# Plot test points
plt.scatter(X_test[:, 0], X_test[:, 1], color='green', marker='x', label='Test Points')

# Plot the separating hyperplane
plt.plot(x, y, 'k-', label='Separating Hyperplane')

# Plot margin boundaries
plt.plot(x, y_margin_up, 'k--', label='Margin Boundaries')
plt.plot(x, y_margin_down, 'k--')

# Mark support vectors
plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Linear SVM: Training Data, Hyperplane, and Margins')
plt.legend()
plt.grid()
plt.show()

# Predict class labels for test points
y_test_pred = svm.predict(X_test)
print("Predicted class labels for test points:")
print(f"x_test1 = {X_test[0]} -> Predicted class label: {y_test_pred[0]}")
print(f"x_test2 = {X_test[1]} -> Predicted class label: {y_test_pred[1]}")
