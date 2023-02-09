import numpy as np
import pickle

from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(5)

def prepare_mnist_train_test():
  digits = datasets.load_digits()

  # Prepare data
  n_samples = len(digits.images)
  train_nparray_x = digits.images.reshape((n_samples, -1))
  train_nparray_y = digits.target

  # Separe data to train and test
  train_x, test_x, train_y, test_y = train_test_split(train_nparray_x, train_nparray_y, stratify=train_nparray_y, test_size=0.25)
  print(f"Training with {len(train_x)} and testing with {len(test_x)}")

  return train_x, test_x, train_y, test_y

def unpickle_model():
  filename = 'finalized_model'
  return pickle.load(open(f"model/{filename}", 'rb'))

def pickle_model(model):
  filename = 'finalized_model'
  pickle.dump(model, open(f"model/{filename}", 'wb'))

def predict_test_model(model_unpickle, test_x, test_y):
  predict = model_unpickle.predict(test_x)
  accuracy = accuracy_score(test_y, predict)
  print(f"Accuracy was {accuracy:.2%}")

def main():
  train_x, test_x, train_y, test_y = prepare_mnist_train_test()
  # Baseline
  dummy = DummyClassifier()
  dummy.fit(train_x, train_y)
  dummy_predict = dummy.predict(test_x)
  baseline_dummy_accuracy = dummy.score(test_y, dummy_predict)
  print(f"Baseline dummy accuracy was {baseline_dummy_accuracy:.2%}")

  #Model training
  model = SVC()
  model.fit(train_x, train_y)
  
  # Save Model
  pickle_model(model)

  # Unpack Model
  model_unpickle = unpickle_model()

  # Test model
  predict_test_model(model_unpickle, test_x, test_y)

if __name__ == "__main__":
  main()