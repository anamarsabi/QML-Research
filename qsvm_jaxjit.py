import os
os.environ["OMP_NUM_THREADS"] = '8'
import pandas as pd
import numpy as np
import pennylane as qml
import tensorflow as tf
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import time
import filelock

#from catalyst import qjit, measure, cond, for_loop, while_loop
import jax
from jax import config
from jax import numpy as jnp
config.update("jax_enable_x64", True)

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, balanced_accuracy_score
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.preprocessing import MaxAbsScaler
# from sklearn.model_selection import KFold

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
  
  return {'SEP_acc': score_sep, 'PPT_acc': score_ppt, 'NPPT_acc': score_nppt}
  
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

# Dimension of the bipartite system
dim = '3x3'
folder = 'efficiency_study'
cat_size = 200
csv_file_path='./results/'+dim+'/qsvm_jaxjit_32pca.csv'
# Number of Principal components to reduce the dataset's dimensionality
n_pca=32

# pennylane works with doubles and tensorflow works with floats.
# We ask tensorflow to work with doubles
tf.keras.backend.set_floatx('float64')

# Initialize a seed for np, so that our results are reproducible
np.random.seed(42)

# We consider training data with a fixed PPT ratio of 0.5, i.e. the 
# proportion of PPT entangled data within the Entangled data of the dataset.
# Reduce the dimensionality of the dataset with Principal Component Analysis
ppt_ratio='1'

parser = ArgumentParser()
parser.add_argument('size')
args = parser.parse_args()
size=args.size

start=time.time()
x_train = np.genfromtxt('./dataset/'+folder+'/train/x_n_'+str(size)+'.csv', delimiter=",",dtype=None)


pca = PCA(n_components = n_pca)
xs_train = pca.fit_transform(x_train)
print(xs_train.shape)

x_test = np.genfromtxt('./dataset/'+dim+'/test/x_test.csv', delimiter=",",dtype=None)
xs_test = pca.transform(x_test)


y_train = np.genfromtxt('./dataset/'+folder+'/train/y_n_'+str(size)+'.csv', delimiter=",",dtype=None)
y_test = np.genfromtxt('./dataset/'+dim+'/test/y_test.csv', delimiter=",",dtype=None)


# Amplitude encoding of 32 variables using 5 qubits (can encode up to 32 inputs)
# Number of qubits of the system
nqubits = 5
# We define a device
dev = qml.device("lightning.qubit", wires = nqubits)

# We define de circuit of our kernel. We use AmplitudeEmbedding which returns an
# operation equivalent to amplitude encoding of the first argument
@qml.qnode(dev, interface="jax")
def kernel_circ(a,b):
    qml.AmplitudeEmbedding(a, wires=range(nqubits), pad_with=0, normalize=True)
    # Computes the adjoint (or inverse) of the amplitude encoding of b
    qml.adjoint(qml.AmplitudeEmbedding(b, wires=range(nqubits), pad_with=0, normalize=True))    # We return an array with the probabilities fo measuring each possible state in the
    # computational basis
    return qml.probs(wires=range(nqubits))


# Now we use the qjit() decorator to jit the above quantum function
start_jit=time.time()
jitted_kernel = jax.jit(kernel_circ)
end_jit=time.time()

# We use SVM of scikit-learn with a custom kernel. 
# This function computes a matrix of kernel evaluations of the data samples,
# known as Gram matrix
def qkernel(A, B):
  return jnp.array([[jitted_kernel(a,b)[0] for b in B] for a in A])

# Metrics scores we will save in a pandas dataframe
acc_scores=[]
f1_scores=[]
prec_scores=[]
rec_scores=[]
bacc_scores=[]
sep_scores=[]
ppt_scores=[]
nppt_scores=[]

# Training of the SVM, compute predictions and metrics 
svm = SVC(kernel = qkernel).fit(xs_train, y_train)
y_train_pred=svm.predict(xs_train)
y_test_pred=svm.predict(xs_test)
end=time.time()

tr_acc = accuracy_score(y_train_pred, y_train)
acc_scores.append(accuracy_score(y_test_pred, y_test))
f1_scores.append(f1_score(y_test, y_test_pred))
prec_scores.append(precision_score(y_test, y_test_pred))
rec_scores.append(recall_score(y_test, y_test_pred))
bacc_scores.append(balanced_accuracy_score(y_test, y_test_pred))
acc_per_type=detailed_accuracy(y_test_pred, cat_size)
sep_scores.append(acc_per_type['SEP_acc'])
ppt_scores.append(acc_per_type['PPT_acc'])
nppt_scores.append(acc_per_type['NPPT_acc'])

time_jit=end_jit-start_jit
print('Jit time: ', time_jit)
print('Training and prediction time: ', end-start)

performance(y_train_pred, y_train, y_test_pred, y_test,cat_size)
# Create a dictionary to sumarise metric's scores
d={'PPT_ratio': ppt_ratio, 'acc_train': tr_acc, 'accuracy': acc_scores, 'f1': f1_scores,
   'precision': prec_scores, 'recall': rec_scores, 'balanced_accuracy': bacc_scores,
   'SEP_accuracy': sep_scores, 'PPT_accuracy': ppt_scores, 'NPPT_accuracy': nppt_scores,
   'time': (end-start)-time_jit, 'time_jit': time_jit  }


lock = filelock.FileLock(csv_file_path+'.lock')
print('Acquiring lock ...')
with lock:
  print('Lock acquired')
  # Create a dictionary to sumarise metric's scores
  d={'size': size, 'PPT_ratio': ppt_ratio, 'acc_train': tr_acc, 'accuracy': acc_scores, 'f1': f1_scores,
    'precision': prec_scores, 'recall': rec_scores, 'balanced_accuracy': bacc_scores,
    'SEP_accuracy': sep_scores, 'PPT_accuracy': ppt_scores, 'NPPT_accuracy': nppt_scores,
    'time': end-start}
  
  # Create a DataFrame with the provided data
  df = pd.DataFrame(data=d)
  with open(csv_file_path, 'a') as csvfile:
    # Append the DataFrame to the CSV file
    df.to_csv(csvfile, header=False, index=False)





# Create a pandas dataframe from the dictionary
#df = pd.DataFrame(data=d)

# Save the dataframe to a CSV file
#df.to_csv('./results/'+dim+'/qsvm_jit_trial.csv', index=False)


