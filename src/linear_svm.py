# test with linear kernel svm
import numpy as np
import pandas as pd
import time

from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, balanced_accuracy_score

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
dim = '4x4'
folder = 'sizetest'
cat_size = 200

# We consider training data with a fixed PPT ratio of 0.5, i.e. the 
# proportion of PPT entangled data within the Entangled data of the dataset.
# Reduce the dimensionality of the dataset with Principal Component Analysis
ppt_ratio='1'
n_pca=128

# Metrics scores we will save in a pandas dataframe
train_acc=[]
acc_scores=[]
f1_scores=[]
prec_scores=[]
rec_scores=[]
bacc_scores=[]
sep_scores=[]
ppt_scores=[]
nppt_scores=[]
times=[]
sizes=np.arange(200,1001,100)
for size in range(200,1001,100):
  x_train = np.genfromtxt('../dataset/'+dim+'/'+folder+'/train/x_n_'+str(size)+'.csv', delimiter=",",dtype=None)
  y_train = np.genfromtxt('../dataset/'+dim+'/'+folder+'/train/y_n_'+str(size)+'.csv', delimiter=",",dtype=None)

  x_test = np.genfromtxt('../dataset/'+dim+'/test/x_test.csv', delimiter=",",dtype=None)
  y_test = np.genfromtxt('../dataset/'+dim+'/test/y_test.csv', delimiter=",",dtype=None)

  pca = PCA(n_components = n_pca)
  xs_train = pca.fit_transform(x_train)
  xs_test = pca.transform(x_test)

  start=time.time()
  svc=SVC(kernel='linear').fit(xs_train,y_train)
  end=time.time()
  training_time=end-start

  y_train_pred=svc.predict(xs_train)
  start=time.time()
  y_test_pred=svc.predict(xs_test)
  end=time.time()
  print('Prediction time: ', end-start)

  prediction_time=end-start

  train_acc.append(accuracy_score(y_train_pred, y_train))
  acc_scores.append(accuracy_score(y_test_pred, y_test))
  f1_scores.append(f1_score(y_test, y_test_pred))
  prec_scores.append(precision_score(y_test, y_test_pred))
  rec_scores.append(recall_score(y_test, y_test_pred))
  bacc_scores.append(balanced_accuracy_score(y_test, y_test_pred))
  acc_per_type=detailed_accuracy(y_test_pred, cat_size)
  sep_scores.append(acc_per_type['SEP_acc'])
  ppt_scores.append(acc_per_type['PPT_acc'])
  nppt_scores.append(acc_per_type['NPPT_acc'])
  times.append(training_time+prediction_time)
  performance(y_train_pred, y_train, y_test_pred, y_test,cat_size)

# Create a dictionary to sumarise metric's scores
d={'size':sizes ,'PPT_ratio': ppt_ratio, 'acc_train': train_acc, 'accuracy': acc_scores, 'f1': f1_scores,
    'precision': prec_scores, 'recall': rec_scores, 'balanced_accuracy': bacc_scores,
    'SEP_accuracy': sep_scores, 'PPT_accuracy': ppt_scores, 'NPPT_accuracy': nppt_scores}

# Create a pandas dataframe from the dictionary
df = pd.DataFrame(data=d)

# Save the dataframe to a CSV file
df.to_csv('../results/'+dim+'/classic_linear_svm_sizetest_128pca.csv', index=False)




