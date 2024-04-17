import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

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
  # Train confusion matrix
  cm = confusion_matrix(y_train, y_train_pred)
  cm_display = ConfusionMatrixDisplay(cm).plot()
  cm_display.ax_.set_title("Train confusion matrix")

  # Test confusion matrix
  cm = confusion_matrix(y_test, y_test_pred)
  cm_display = ConfusionMatrixDisplay(cm).plot()
  cm_display.ax_.set_title("Test confusion matrix")
