from sklearn.naive_bayes import MultinomialNB
from collections import Counter
import pandas as pd
dataframe = pd.read_csv('buscas.csv')

X_df, Y_df = dataframe[['home', 'busca', 'logado']], dataframe['comprou']

x_dummies_df = pd.get_dummies(X_df)
y_dummies_df = Y_df

X = x_dummies_df.values
Y = y_dummies_df.values

porcentagem_treino = 0.9

tamanho_treino = int(porcentagem_treino * len(Y))
tamanho_teste = len(Y) - tamanho_treino

treino_dados = X[:tamanho_treino]
treino_marcacoes = Y[:tamanho_treino]

teste_dados = X[-tamanho_teste:]
teste_marcacoes = Y[-tamanho_teste:]

modelo = MultinomialNB()

modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(teste_dados)
acertos = (resultado == teste_marcacoes)

total_acertos = sum(acertos)
total_elementos = len(teste_dados)

taxa_acerto = 100.0 * total_acertos / total_elementos

print("============================================")
print("Taxa de acerto do algoritmo: %f" % taxa_acerto)
print("Total de elementos testados: %d" % total_elementos)

# a eficacia do algoritimo que chuta tudo um único valor
acerto_base = max(Counter(teste_marcacoes).values())
taxa_de_acerto_base = 100.0 * acerto_base / len(teste_marcacoes)

print("============================================")
print("Taxa de acerto base: %f" % taxa_de_acerto_base)