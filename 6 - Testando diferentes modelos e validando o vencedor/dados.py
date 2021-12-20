import csv


def carregar_buscas():
    X, Y = [], []
    arquivo = open('buscas.csv', 'r')
    leitor = csv.reader(arquivo)
    leitor.__next__()

    for home, busca, logado, comprou in leitor:
        X.append([int(home), busca, int(logado)])
        Y.append(int(comprou))

    return X, Y
