# é gordo
# tem perna curta
# late?

from sklearn.naive_bayes import MultinomialNB

# Criando objeto de porcos
porco1 = [1, 1, 0]
porco2 = [1, 1, 0]
porco3 = [1, 1, 0]

# Criando objeto de cachorros
cachorro1 = [1, 1, 1]
cachorro2 = [0, 1, 1]
cachorro3 = [0, 1, 1]

# Dados para treinamento do algoritimo
dados = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]

# Resultado esperado
marcacoes = [1, 1, 1, -1, -1, -1]

# Criando modelo de algoritmo
modelo = MultinomialNB()

# Treinando algoritimo com os dados montados
modelo.fit(dados, marcacoes)

# objeto para teste do algoritimo
misterioso1 = [1, 1, 1]
misterioso2 = [1, 0, 0]
misterioso3 = [0, 0, 1]

testes = [misterioso1, misterioso2, misterioso3]

marcacoes_testes = [-1, 1, -1]

# Prevendo qual classe pertence o objeto 'misterioso'
resultado = modelo.predict(testes)

# printando o resultado
print(resultado)

diferencas = resultado - marcacoes_testes
print(diferencas)

acertos = [d for d in diferencas if d == 0]

print(acertos)

total_acertos = len(acertos)
total_elementos = len(testes)

taxa_acerto = 100.0 * total_acertos / total_elementos

print(taxa_acerto)

