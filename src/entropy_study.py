import pandas as pd
import numpy as np
import pennylane as qml
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
from sklearn.decomposition import PCA


# pennylane works with doubles and tensorflow works with floats.
# We ask tensorflow to work with doubles

tf.keras.backend.set_floatx('float64')

# We set a seed for the packages so the results are reproducible
seed=4321
np.random.seed(seed)
tf.random.set_seed(seed)

# Dimension of the bipartite system
dim = '3x3'
cat_size = 1000

# Number of Principal components to reduce the dataset's dimensionality
n_pca=32

x_train = np.genfromtxt('./dataset/'+dim+'/train/x_1.csv', delimiter=",",dtype=None)
x_test = np.genfromtxt('./dataset/'+dim+'/test/x_test.csv', delimiter=",",dtype=None)


pca = PCA(n_components = n_pca)
xs_train = pca.fit_transform(x_train)
xs_test = pca.transform(x_test)


y_train = np.genfromtxt('./dataset/'+dim+'/train/y_train.csv', delimiter=",",dtype=None)
y_test = np.genfromtxt('./dataset/'+dim+'/test/y_test.csv', delimiter=",",dtype=None)

print('train shape ', x_train.shape)

# Amplitude encoding of 64 variables using 5 qubits (can encode up to 32 inputs)
# Number of qubits of the system
nqubits = 5
# We define a device
dev = qml.device("default.qubit", wires = nqubits)

# Define the array
elements = [0, 1, 2, 3, 4]

# Create the combinations
combinations = [(i,) for i in elements] + list(itertools.combinations(elements, 2)) + list(itertools.combinations(elements, 3)) + list(itertools.combinations(elements, 4))

# We define de circuit of our kernel. We use AmplitudeEmbedding which returns an
# operation equivalent to amplitude encoding of the first argument
@qml.qnode(dev)
def encode(a):
    x=qml.AmplitudeEmbedding(a, wires=range(nqubits), pad_with=0, normalize=True)
    # x=qml.state()
    # print(x)
    return qml.state()


entropies=[]
entropies_base2=[]
for matrix in xs_train:
    x=encode(matrix)
    aux_entropies=[]
    aux_base2=[]
    for combo in combinations:
        aux_entropies.append(qml.math.vn_entropy(x, indices=list(combo)))
        aux_base2.append(qml.math.vn_entropy(x, indices=list(combo), base=2))
    entropies.append(np.mean(aux_entropies))
    entropies_base2.append(np.mean(aux_base2))

#print(entropies)
mean_sep_entropy=np.mean(entropies[:cat_size])
mean_ppt_entropy=np.mean(entropies[cat_size+1:cat_size+int(cat_size/2)])
mean_nppt_entropy=np.mean(entropies[cat_size+int(cat_size/2):2*cat_size])

print('Mean separable entropy: ', mean_sep_entropy)
print('Mean PPT-ent entropy: ', mean_ppt_entropy)
print('Mean NPPT-ent entropy', mean_nppt_entropy)