# import configparser as cp
# import pandas as pd
# import numpy as np
# from time import time
# from sklearn.metrics import confusion_matrix
# from tensorflow.keras.callbacks import TensorBoard
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, CuDNNLSTM
# from keras.optimizers import Adam, SGD

# nb_cellule_cachee = 128
# nb_param_sortie = 5
# fonction_activation = 'sigmoid'
# fonction_taux_perte = 'mean_squared_error'
# nb_couche_cachee = 2

# modele = Sequentiel()

# modele.ajouter(LSTM(cellule_LSTM=128,
#                     forme=(param_entree=8, echantillon=494021)))
# modele.ajouter(Perte(20 %))

# modele.ajouter(Liason(param_sortie=5,
#                       fonction_activation='sigmoide'))

# modele.compiler(taux_perte=erreur_quadratique_moyenne,
#                 taux_apprentissage=0.001,
#                 mesure_efficacite=['précision'])

# modele.entrainement(x=entree_entrainement, y=resultat_entrainement,
#                     nb_iteration=20,
#                     validation=(x=entree_test, y=resultat_test))
