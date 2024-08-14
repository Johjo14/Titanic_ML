#Création d'une petite fonction pour extraire la première lettre de la colonne cabine
import pandas as pd

def extract_first_letter(serie):
  #Récupère une Serie en argument
  #Retourne une DataFrame (question de compatibilité pour le col_transformer)
  return pd.DataFrame(serie.str[0])