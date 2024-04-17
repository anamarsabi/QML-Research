import pandas as pd
import numpy as np
import pennylane as qml
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import KFold

#from src.aux_output import performance
def detailed_accuracy(y_test_pred, size):
  test_size=size
  pred_split=np.array_split(y_test_pred,3)
  y_sep_pred=pred_split[0]
  y_ppt_pred=pred_split[1]
  y_nppt_pred=pred_split[2]

  score_sep = accuracy_score(y_sep_pred, np.full(test_size, 0))
  score_ppt= accuracy_score(y_ppt_pred, np.full(test_size,1))
  score_nppt= accuracy_score(y_nppt_pred, np.full(test_size,1))

  print("SEP accuracy: ", score_sep)
  print("PPT accuracy: ", score_ppt)
  print("NPPT accuracy: ", score_nppt)
  
def performance(y_train_pred, y_train, y_test_pred, y_test, size):
  tr_acc = accuracy_score(y_train_pred, y_train)
  test_acc = accuracy_score(y_test_pred, y_test)

  tr_f1 = f1_score(y_train, y_train_pred)
  test_f1 = f1_score(y_test, y_test_pred)

  print("Train accuracy: ", tr_acc)
  print("Train F-1 score: ", tr_f1)

  print("\nTest accuracy: ", test_acc)
  print("Test F-1 score: ", test_f1)

  # Test accuracy per type
  print("\nTest accuracy broken down per type")
  detailed_accuracy(y_test_pred, size)

  print("\n")
  # Train confusion matrix
  cm = confusion_matrix(y_train, y_train_pred)
  cm_display = ConfusionMatrixDisplay(cm).plot()
  cm_display.ax_.set_title("Train confusion matrix")

  # Test confusion matrix
  cm = confusion_matrix(y_test, y_test_pred)
  cm_display = ConfusionMatrixDisplay(cm).plot()
  cm_display.ax_.set_title("Test confusion matrix")


# Dimension of the bipartite system
dim = 'toy'

# Number of Principal components to reduce the dataset's dimensionality
n_pca=20

# pennylane works with doubles and tensorflow works with floats.
# We ask tensorflow to work with doubles
tf.keras.backend.set_floatx('float64')

# Initialize a seed for np, so that our results are reproducible
np.random.seed(42)


# We consider training data with a fixed PPT ratio of 0.5, i.e. the 
# proportion of PPT entangled data within the Entangled data of the dataset.
# Reduce the dimensionality of the dataset with Principal Component Analysis
ppt_ratio='05'


x_train = np.genfromtxt('./dataset/'+dim+'/train/x_'+ppt_ratio+'.csv', delimiter=",",dtype=None)
print(x_train.shape)

pca = PCA(n_components = n_pca)
xs_train = pca.fit_transform(x_train)
print(xs_train.shape)

x_test = np.genfromtxt('./dataset/'+dim+'/test/x_test.csv', delimiter=",",dtype=None)
xs_test = pca.transform(x_test)

y_train = np.genfromtxt('./dataset/'+dim+'/train/y_train.csv', delimiter=",",dtype=None)
y_test = np.genfromtxt('./dataset/'+dim+'/test/y_test.csv', delimiter=",",dtype=None)


# Amplitude encoding of 64 variables using 5 qubits (can encode up to 32 inputs)
# Number of qubits of the system
nqubits = 5
# We define a device
dev = qml.device("default.qubit", wires = nqubits)

# We define de circuit of our kernel. We use AmplitudeEmbedding which returns an
# operation equivalent to amplitude encoding of the first argument
@qml.qnode(dev)
def kernel_circ(a,b):
    qml.AmplitudeEmbedding(a, wires=range(nqubits), pad_with=0, normalize=True)
    # Computes the adjoint (or inverse) of the amplitude encoding of b
    qml.adjoint(qml.AmplitudeEmbedding(b, wires=range(nqubits), pad_with=0, normalize=True))    # We return an array with the probabilities fo measuring each possible state in the
    # computational basis
    return qml.probs(wires=range(nqubits))

def qkernel(A, B):
  return np.array([[kernel_circ(a,b)[0] for b in B] for a in A])


svm = SVC(kernel = qkernel).fit(xs_train, y_train)
y_train_pred=svm.predict(xs_train)
y_test_pred=svm.predict(xs_test)

performance(y_train_pred, y_train, y_test_pred, y_test,10)

# tr_acc=accuracy_score(y_train_pred, y_train)
# print("Train accuracy: ", tr_acc)
# test_acc=accuracy_score(y_test_pred, y_test)
# print("Test accuracy: ", test_acc)
# # Test confussion matrix
# cm = confusion_matrix(y_test, y_test_pred)
# # Test accuracy per type
# detailed_accuracy(y_test_pred, 1000)


