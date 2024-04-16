import pandas as pd
import numpy as np
import pennylane as qml
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, balanced_accuracy_score
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.preprocessing import MaxAbsScaler
# from sklearn.model_selection import KFold

from src.aux_output import detailed_accuracy, performance

# Dimension of the bipartite system
dim = '3x3'

# Number of Principal components to reduce the dataset's dimensionality
n_pca=32

# pennylane works with doubles and tensorflow works with floats.
# We ask tensorflow to work with doubles
tf.keras.backend.set_floatx('float64')

# Initialize a seed for np, so that our results are reproducible
np.random.seed(42)


# We consider an array of training data with different PPT ratio, i.e. the 
# proportion of PPT entangled data within the Entangled data of the dataset.
# Reduce the dimensionality of the dataset with Principal Component Analysis
ppt_ratio=['025', '05', '075', '1']
training_data=[]
pca = PCA(n_components = n_pca)
for r in ppt_ratio:
    xs_train = pca.fit_transform(np.genfromtxt('./dataset/'+dim+'/train/x_'+r+'.csv', delimiter=",",dtype=None))
    training_data.append(xs_train)

x_test = np.genfromtxt('./dataset/'+dim+'/test/x_test.csv', delimiter=",",dtype=None)
xs_test = pca.transform(x_test)

y_train = np.genfromtxt('./dataset/'+dim+'/train/y_train.csv', delimiter=",",dtype=None)
y_test = np.genfromtxt('./dataset/'+dim+'/test/y_test.csv', delimiter=",",dtype=None)


# Amplitude encoding of 64 variables using 5 qubits (can encode up to 32 inputs)
# Number of qubits of the system
nqubits = 5
# We define a device
dev = qml.device("lightning.qubit", wires = nqubits)

# We define de circuit of our kernel. We use AmplitudeEmbedding which returns an
# operation equivalent to amplitude encoding of the first argument
@qml.qnode(dev)
def kernel_circ(a,b):
    qml.AmplitudeEmbedding(a, wires=range(nqubits), pad_with=0, normalize=True)
    # Computes the adjoint (or inverse) of the amplitude encoding of b
    qml.adjoint(qml.AmplitudeEmbedding(b, wires=range(nqubits), pad_with=0, normalize=True))    # We return an array with the probabilities fo measuring each possible state in the
    # computational basis
    return qml.probs(wires=range(nqubits))

# We use SVM of scikit-learn with a custom kernel. 
# This function computes a matrix of kernel evaluations of the data samples,
# known as Gram matrix
def qkernel(A, B):
  return np.array([[kernel_circ(a,b)[0] for b in B] for a in A])

# Metrics scores we will save in a pandas dataframe
acc_scores=[]
f1_scores=[]
prec_scores=[]
rec_scores=[]
bacc_scores=[]
sep_scores=[]
ppt_scores=[]
nppt_scores=[]

# Training of the SVM, compute predictions and metrics for each PPT ratio training set 
for xs_train in training_data:
    svm = SVC(kernel = qkernel).fit(xs_train, y_train)
    y_train_pred=svm.predict(xs_train)
    y_test_pred=svm.predict(xs_test)
    
    tr_acc = accuracy_score(y_train_pred, y_train)
    acc_scores.append(accuracy_score(y_test_pred, y_test))
    f1_scores.append(f1_score(y_test, y_test_pred))
    prec_scores.append(precision_score(y_test, y_test_pred))
    rec_scores.append(recall_score(y_test, y_test_pred))
    bacc_scores.append(balanced_accuracy_score(y_test, y_test_pred))
    acc_per_type=detailed_accuracy(y_test_pred, 1000)
    sep_scores.append(acc_per_type['SEP_acc'])
    ppt_scores.append(acc_per_type['PPT_acc'])
    nppt_scores.append(acc_per_type['NPPT_acc'])

# Create a dictionary to sumarise metric's scores
d={'PPT_ratio': ppt_ratio, 'acc_train': tr_acc, 'accuracy': acc_scores, 'f1': f1_scores,
   'precision': prec_scores, 'recall': rec_scores, 'balanced_accuracy': bacc_scores,
   'SEP_accuracy': sep_scores, 'PPT_accuracy': ppt_scores, 'NPPT_accuracy': nppt_scores}

# Create a pandas dataframe from the dictionary
df = pd.DataFrame(data=d)

# Save the dataframe to a CSV file
df.to_csv('./results/qsvm_results.csv', index=False)


# tr_acc=accuracy_score(y_train_pred, y_train)
# print("Train accuracy: ", tr_acc)
# test_acc=accuracy_score(y_test_pred, y_test)
# print("Test accuracy: ", test_acc)
# # Test confussion matrix
# cm = confusion_matrix(y_test, y_test_pred)



