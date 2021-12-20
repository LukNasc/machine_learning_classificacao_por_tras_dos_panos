from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from collections import Counter
import pandas as pd

# teste inicial: home, busca, logado => comprou
# home,  busca
# home, logado
# busca, logado
# busca: 85.71% (7 testes)


def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):
    modelo.fit(treino_dados, treino_marcacoes)

    resultado = modelo.predict(teste_dados)
    acertos = (resultado == teste_marcacoes)

    total_acertos = sum(acertos)
    total_elementos = len(teste_dados)

    taxa_acerto = 100.0 * total_acertos / total_elementos

    print("============================================")
    print("Taxa de acerto do algoritmo {algoritimo}: {taxa_acerto}".format(
        algoritimo=nome, taxa_acerto=taxa_acerto))
    print("Total de elementos testados: %d" % total_elementos)

    return taxa_acerto


dataframe = pd.read_csv('buscas2.csv')

X_df, Y_df = dataframe[['home', 'busca', 'logado']], dataframe['comprou']

x_dummies_df = pd.get_dummies(X_df)
y_dummies_df = Y_df

X = x_dummies_df.values
Y = y_dummies_df.values

porcentagem_treino = 0.8
porcentagem_teste = 0.1

tamanho_treino = int(porcentagem_treino * len(Y))
tamanho_teste = int(porcentagem_teste * len(Y))
tamanho_validacao = len(Y) - tamanho_teste - tamanho_treino

treino_dados = X[0:tamanho_treino]
treino_marcacoes = Y[0:tamanho_treino]

fim_teste = tamanho_treino + tamanho_teste
teste_dados = X[tamanho_treino:fim_teste]
teste_marcacoes = Y[tamanho_treino:fim_teste]

validacao_dados = X[fim_teste:]
validacao_marcacoes = Y[fim_teste:]

multinomial_modelo = MultinomialNB()
adaboost_modelo = AdaBoostClassifier()

resultado_muiltinomial = fit_and_predict(nome="MultinomialNB", modelo=multinomial_modelo, treino_dados=treino_dados,
                                         treino_marcacoes=treino_marcacoes, teste_dados=teste_dados, teste_marcacoes=teste_marcacoes)

resultado_adaboost = fit_and_predict(nome="AdaBoostClassifier", modelo=adaboost_modelo, treino_dados=treino_dados,
                                     treino_marcacoes=treino_marcacoes, teste_dados=teste_dados, teste_marcacoes=teste_marcacoes)

if resultado_muiltinomial > resultado_adaboost:
    vencedor = multinomial_modelo
else:
    vencedor = adaboost_modelo

resultado = vencedor.predict(validacao_dados)
acertos = (resultado == validacao_marcacoes)

total_acertos = sum(acertos)
total_elementos = len(validacao_marcacoes)
taxa_acerto = 100.0 * total_acertos / total_elementos

print("============================================")
print("Taxa de acerto do vencedor no mundo real: {taxa_acerto}".format(
    taxa_acerto=taxa_acerto))

# a eficacia do algoritimo que chuta tudo um único valor
acerto_base = max(Counter(validacao_marcacoes).values())
taxa_de_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)

print("============================================")
print("Taxa de acerto base: %f" % taxa_de_acerto_base)
