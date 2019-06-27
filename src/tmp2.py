modele = Sequentiel()

modele.ajouter(LSTM(cellule_LSTM=128,
                    forme=(param_entree=8, echantillon=494021)))
modele.ajouter(Perte(20 %))

modele.ajouter(Liason(param_sortie=5,
                      fonction_activation='sigmoide'))

modele.compiler(taux_perte=erreur_quadratique_moyenne,
                taux_apprentissage=0.001,
                mesure_efficacite=['précision'])

modele.entrainement(x=entree_entrainement, y=resultat_entrainement,
                    nb_iteration=20,
                    validation=(x=entree_test, y=resultat_test))
