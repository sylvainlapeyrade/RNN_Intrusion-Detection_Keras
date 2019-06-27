import numpy as np
import matplotlib.pyplot as plt


y = [0.026684, 0.025450, 0.025348, 0.024538, 0.024914, 0.023963]
x = ['Encodeur', 'Optimiseur', 'Fonction \n d\'activation',
     'unit_nb', 'layer_nb', 'dropout']
z = ['Standarscaler', 'rmsprop', 'sigmoid', 128, 1, 0.1]
plt.plot(x, y, color='blue')
plt.title("Évolution du taux de perte en validation")
plt.ylabel("Taux d'erreur", fontsize=12, color='green')
plt.xlabel("Paramètre", fontsize=12, color='green')
plt.show()


# y1 = np.array([0.054, 0.028, 0.020, 0.016, 0.015,
#  0.014, 0.013, 0.013, 0.012, 0.011])
# y2 = np.array([0.059, 0.033, 0.025, 0.021, 0.020,
#  0.018, 0.017, 0.020, 0.023, 0.030])

# plt.plot(y1, color = 'blue', label="entrainement")
# plt.plot(y2, color = 'red', label="validation")
# plt.title("Apprentissage")
# plt.legend()
# plt.ylabel("Taux d'erreur")
# plt.xlabel("Temps")
# plt.show() # affiche la figure a l'ecran

# # green for heaviside
# in_array = np.linspace(-2, 2, 20)
# out_array = np.heaviside(in_array, 0.5)
# plt.plot(in_array, out_array, color = 'green', marker = "o")
# plt.title("Heaviside")
# plt.show()

# # red for tanh
# in_array = np.linspace(-2, 2, 20)
# out_array = np.tanh(in_array)
# plt.plot(in_array, out_array, color = 'red', marker = "o")
# plt.title("Tangente hyperbolique")
# plt.show()

# # Blue for Sigmoid
# in_array2 = np.linspace(-2, 2, 20)
# out_array2 = 1 / (1 + np.exp(-in_array2))
# plt.plot(in_array2, out_array2, color = 'blue', marker = "o")
# plt.title("Sigmoïde")
# plt.show()
