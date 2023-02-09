import json

import numpy as np
import requests
from mnist_bentoml import prepare_mnist_train_test

SERVICE_ENDPOINT = "http://localhost:3000/classify"

def get_mnist_data():
  _, test_x, _, test_y = prepare_mnist_train_test()
  return test_x[:7], test_y[:7]

def request_bento_service(service_endpoint, test_x):
  test_x_json = json.dumps(test_x.tolist())
  response = requests.post(
    service_endpoint,
    data=test_x_json,
    headers={'Content-Type': 'application/json'}
  )
  return response.text

def main():
  test_x, test_y = get_mnist_data()
  predict = request_bento_service(SERVICE_ENDPOINT, test_x)

  print(f"Predict: {predict}")
  print(f"Expected: {test_y}")

if __name__ == "__main__":
  main()