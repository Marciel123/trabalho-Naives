from sklearn.ensemble import RandomForestClassifier
import pandas as pd #manipulacao de dados



dados=pd.read_csv('/content/drive/MyDrive/inteligencia artificial/trabalho iris/Iris.csv')

dados.head()
classes = dados['Species']
nomesColunas = dados.columns.to_list()
tamanho = len(nomesColunas)#quantos nomes tem
nomesColunas = nomesColunas[1:tamanho-1] #retira o ultimo
features = dados[nomesColunas] #monta o features



from sklearn.model_selection import train_test_split

features_treino,features_teste,classes_treino,classes_teste = train_test_split(features,
                                                                               classes,
                                                                               test_size=0.6,
                                                                               random_state=2)
#criando a floresta
floresta = RandomForestClassifier(n_estimators=1000) #constroi a floresta

#treinar a floresta

floresta.fit(features_treino,classes_treino)

#testar quanto a floresta acerta
predicoes = floresta.predict(features_teste)
#verificar quanto das predicoes foram acertos
from sklearn import metrics

print(metrics.classification_report(classes_teste,predicoes))
