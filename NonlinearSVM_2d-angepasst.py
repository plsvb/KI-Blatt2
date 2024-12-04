"""
================================================
SVM: Maximum margin separating decision boundary
================================================

Plot the maximum margin separating decision boundary within a two-class
separable dataset using a Support Vector Machines classifier.
"""
print(__doc__)

import numpy as np
import pylab as pl
from sklearn import svm

# Load training data
X = np.loadtxt('svm_training_inputs.txt')         # Training data
tmp = np.loadtxt('svm_training_targets.txt')      # Training targets
Y = np.array([[tmp[i]] for i in range(tmp.size)])
Y.shape = tmp.size

# Load test data
X_test = np.loadtxt('svm_test_inputs.txt') 

# Create a mesh to plot in
h = 0.02  # Step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Fit the model (First Kernel: RBF)
print("Training SVM with RBF kernel (gamma=0.5, C=1)...")
clf = svm.SVC(kernel='rbf', C=1, gamma=0.5)  # Radial Basis Function Kernel
clf.fit(X, Y)

# Classify test patterns
Y_test = clf.predict(X_test)
print("Kernel: RBF (gamma=0.5, C=1)")
print("Classification results for the test patterns:")
print(Y_test)

# Plot the decision boundary
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
pl.set_cmap(pl.cm.Paired)  # Sets colormap
pl.contourf(xx, yy, Z)

# Plot training points
pl.scatter(X[:, 0], X[:, 1], edgecolor='black', c=Y, marker='o', s=20)

# Plot test points
pl.scatter(X_test[:, 0], X_test[:, 1], edgecolor='black', c=Y_test, marker='o', s=80)

pl.title('SVM data (RBF kernel)')
pl.xlabel('x1')
pl.ylabel('x2')
pl.axis('tight')
pl.show()

# Fit the model (Second Kernel: Polynomial)
print("Training SVM with Polynomial kernel (degree=3, C=1, gamma='scale')...")
clf = svm.SVC(kernel='poly', C=1, degree=3, gamma='scale')  # Polynomial Kernel
clf.fit(X, Y)

# Classify test patterns
Y_test = clf.predict(X_test)
print("Kernel: Polynomial (degree=3, C=1, gamma='scale')")
print("Classification results for the test patterns:")
print(Y_test)

# Plot the decision boundary
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
pl.set_cmap(pl.cm.Paired)  # Sets colormap
pl.contourf(xx, yy, Z)

# Plot training points
pl.scatter(X[:, 0], X[:, 1], edgecolor='black', c=Y, marker='o', s=20)

# Plot test points
pl.scatter(X_test[:, 0], X_test[:, 1], edgecolor='black', c=Y_test, marker='o', s=80)

pl.title('SVM data (Polynomial kernel)')
pl.xlabel('x1')
pl.ylabel('x2')
pl.axis('tight')
pl.show()
