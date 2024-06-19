import numpy as np
import pandas as pd
import time
import fcntl

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, balanced_accuracy_score
from argparse import ArgumentParser
import filelock

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
cat_size = 200

folder = 'sizetest'
csv_file_path = '../results/'+dim+'/classic_sizetest.csv'

# We consider training data with a fixed PPT ratio of 0.5, i.e. the 
# proportion of PPT entangled data within the Entangled data of the dataset.
# Reduce the dimensionality of the dataset with Principal Component Analysis
ppt_ratio='1'

parser = ArgumentParser()
parser.add_argument('size')
args = parser.parse_args()
size=args.size

x_train = np.genfromtxt('../dataset/'+dim+'/'+folder+'/train/x_n_'+str(size)+'.csv', delimiter=",",dtype=None)
x_test = np.genfromtxt('../dataset/'+dim+'/test/x_test.csv', delimiter=",",dtype=None)

y_train = np.genfromtxt('../dataset/'+dim+'/'+folder+'/train/y_n_'+str(size)+'.csv', delimiter=",",dtype=None)
y_test = np.genfromtxt('../dataset/'+dim+'/test/y_test.csv', delimiter=",",dtype=None)


# SVM
model_type = SVC()
model_params = [
    {'kernel': ['poly'], 'degree': np.arange(2, 7), 'gamma': np.logspace(-5, 5, num=5), 'C': np.logspace(-5, 5, num=5)},
    {'kernel': ['rbf'], 'gamma': np.logspace(-5, 5, num=5), 'C': np.logspace(-5, 5, num=5)},
    {'kernel': ['sigmoid'], 'gamma': np.logspace(-5, 5, num=5), 'C': np.logspace(-5, 5, num=5)}]


n_models = 1
# TRAINING
models = []

training_times=[]
for i in range(n_models) :
    # TRAINING SET CREATION
    start=time.time()
    print(f'---- Training model ({i})...')
    model = GridSearchCV(model_type, model_params, cv=StratifiedKFold(shuffle=True)).fit(x_train, y_train)
    end=time.time()
    print('training results :')
    print(model.best_params_)
    print(model.best_score_)
    print('Training time: ', end-start)
    training_times.append(end-start)
    models.append(model)

# Metrics scores we will save in a pandas dataframe
tr_accs= []
acc_scores=[]
f1_scores=[]
prec_scores=[]
rec_scores=[]
bacc_scores=[]
sep_scores=[]
ppt_scores=[]
nppt_scores=[]

prediction_times=[]

for i in range(len(models)) :
    print('---- Test Model ', i)
    y_train_pred=models[i].predict(x_train)
    start=time.time()
    y_test_pred=models[i].predict(x_test)
    end=time.time()
    print('Prediction time: ', end-start)
    prediction_times.append(end-start)

    tr_accs.append(accuracy_score(y_train_pred, y_train))
    acc_scores.append(accuracy_score(y_test_pred, y_test))
    f1_scores.append(f1_score(y_test, y_test_pred))
    prec_scores.append(precision_score(y_test, y_test_pred))
    rec_scores.append(recall_score(y_test, y_test_pred))
    bacc_scores.append(balanced_accuracy_score(y_test, y_test_pred))
    acc_per_type=detailed_accuracy(y_test_pred, cat_size)
    sep_scores.append(acc_per_type['SEP_acc'])
    ppt_scores.append(acc_per_type['PPT_acc'])
    nppt_scores.append(acc_per_type['NPPT_acc'])

    performance(y_train_pred, y_train, y_test_pred, y_test,cat_size)

times=training_times+prediction_times
lock = filelock.FileLock(csv_file_path+'.lock')
print('Acquiring lock ...')
with lock:
  print('Lock acquired')
  # Create a dictionary to sumarise metric's scores
  d={'size': size, 'PPT_ratio': ppt_ratio, 'acc_train': tr_accs, 'accuracy': acc_scores, 'f1': f1_scores,
    'precision': prec_scores, 'recall': rec_scores, 'balanced_accuracy': bacc_scores,
    'SEP_accuracy': sep_scores, 'PPT_accuracy': ppt_scores, 'NPPT_accuracy': nppt_scores,
    'time': end-start}
  
  # Create a DataFrame with the provided data
  df = pd.DataFrame(data=d)
  with open(csv_file_path, 'a') as csvfile:
    # Append the DataFrame to the CSV file
    df.to_csv(csvfile, header=False, index=False)

