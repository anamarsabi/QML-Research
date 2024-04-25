import pennylane as qml

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time 

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Dimension of the bipartite system
folder = 'efficiency_study'

# Number of Principal components to reduce the dataset's dimensionality
n_pca=32

# pennylane works with doubles and tensorflow works with floats.
# We ask tensorflow to work with doubles
tf.keras.backend.set_floatx('float64')

# Initialize a seed for np, so that our results are reproducible
np.random.seed(42)

# Amplitude encoding of 64 variables using 5 qubits (can encode up to 32 inputs)
# Number of qubits of the system
nqubits = 5
# We define a device
dev = qml.device("default.qubit", wires = nqubits, batch_obs=True)

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


sizes=np.arange(100, 1001, 100)
pca_times=[]
training_times=[]
prediction_times=[]
total_times=[]
accuracy_n=[]

size=100
start=time.time()

x_train = np.genfromtxt('./dataset/'+folder+'/train/x_n_'+str(size)+'.csv', delimiter=",",dtype=None)
x_test = np.genfromtxt('./dataset/'+folder+'/test/x_test.csv', delimiter=",",dtype=None)

start_pca = time.time()

pca = PCA(n_components = n_pca)
xs_train = pca.fit_transform(x_train)
xs_test = pca.transform(x_test)

end_pca=time.time()
pca_times.append(end_pca - start_pca)


y_train = np.genfromtxt('./dataset/'+folder+'/train/y_n_'+str(size)+'.csv', delimiter=",",dtype=None)
y_test = np.genfromtxt('./dataset/'+folder+'/test/y_test.csv', delimiter=",",dtype=None)

start_train=time.time()
svm = SVC(kernel = qkernel).fit(xs_train, y_train)
end_train=time.time()
training_times.append(end_train - start_train)

start_test=time.time()
y_test_pred=svm.predict(xs_test)
end_test=time.time()
prediction_times.append(end_test-start_test)

test_acc = accuracy_score(y_test_pred, y_test)
end=time.time()
total_times.append(end - start)

accuracy_n.append(test_acc)
#print("Test accuracy for n = {} : {}".format(size, test_acc))
print("#### n=100, ONLY default.qubit")
print("Total time: ", total_times)
print("PCA time: ", pca_times)
print("Training_time: ",training_times)
print("Predicition_time: ", prediction_times)
print("\n")

# # Create a dictionary to save times
# d={'size':sizes,'total_time': total_times,'pca_time': pca_times, 'training_time': training_times,
#    'prediction_time': prediction_times, 'accuracy': accuracy_n,}

# # Create a pandas dataframe from the dictionary
# df = pd.DataFrame(data=d)

# # Save the dataframe to a CSV file
# df.to_csv('./results/efficiency_results.csv', index=False)