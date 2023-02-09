import bentoml
import pickle

def save_model_in_bento():
  filename = 'finalized_model'
  model = pickle.load(open(f"model/{filename}", 'rb'))
  bento_model = bentoml.sklearn.save_model("sklearn_model", model)
  print(f"Bento model tag: {bento_model.tag}")

if __name__ == '__main__':
  save_model_in_bento()