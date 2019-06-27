import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

value = 'activation_fct'
name = 'Fonction d\'activation'

df = pd.read_csv("./results2/"+value+".csv", index_col=False)

names = df[value].unique().tolist()

df_loss = pd.DataFrame(columns=names)
for i in range(len(names)):
    df_encoder_loss = df.loc[df[value] == names[i]]
    df_encoder_loss = df_encoder_loss.nsmallest(5, 'val_loss')
    df_loss[names[i]] = np.array(df_encoder_loss['val_loss'])
print(df_loss.mean())
print(df_loss.mean().idxmin())
print(df_loss.mean().min())

boxplot_loss = df_loss.boxplot(column=names, figsize=(10, 7))

boxplot_loss.axes.set_title(
    '5 meilleurs taux de perte par '+name, fontsize=20)
boxplot_loss.set_xlabel(name, fontsize=14)
boxplot_loss.set_ylabel('Taux d\'erreur en validation', fontsize=14)
boxplot_loss.figure.savefig('./results2/'+value+'_loss.jpg',
                            format='png')

plt.show()

df_acc = pd.DataFrame(columns=names)
for i in range(len(names)):
    df_encoder_acc = df.loc[df[value] == names[i]]
    df_encoder_acc = df_encoder_acc.nlargest(5, 'val_acc')
    df_acc[names[i]] = np.array(df_encoder_acc['val_acc'])

boxplot_acc = df_acc.boxplot(column=names, figsize=(10, 7))

boxplot_acc.axes.set_title(
    '5 meilleures précisions par '+name, fontsize=20)
boxplot_acc.set_xlabel(name, fontsize=14)
boxplot_acc.set_ylabel('Taux de précision en validation', fontsize=14)
boxplot_acc.figure.savefig('./results2/'+value+'_acc.jpg',
                           format='png')
plt.show()
