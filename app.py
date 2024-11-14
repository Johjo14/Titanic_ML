import sklearn
import joblib
import pandas as pd
import numpy as np
from fonctions_extracLetter import extract_first_letter
from xgboost import XGBClassifier
from flask import Flask, request


#Serveur de developpement
#Chargement de du modèle
pipeline = joblib.load("xgb_model_titanic.pkl")
print(pipeline)

#Démarrer l'application Flask
app = Flask("__name__")

#Faire des predictions
@app.route("/predict", methods=["POST"])
def predict():
  df_titanic = pd.DataFrame(request.json)
  resultat = pipeline.predict(df_titanic)[0] #Index 0 pour récuprer le nombre à l'interieur
  return(str(resultat), 201)
  

#Tester l'API(ping)
@app.route("/ping", methods = ["GET"])
def ping():
  return("pong", 200)

#Création d'une page d'accueil
@app.route("/")
def index():
  return "<h1>Bienvenu dans l'API pour la classification des passagers du Titanic.</h1>"

#Si on est dans le 'main', on lance.
if __name__ == "__main__":
  app.run(host="0.0.0.0", port=8080)
