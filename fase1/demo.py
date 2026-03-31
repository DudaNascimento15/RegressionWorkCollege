import matplotlib.pyplot as plt
import numpy as np
import os
import requests

dataset = 0
x = []
y = []

def ler_arquivo():
    global x, y, dataset
    file_path = 'datasetFase1.txt'
    url = 'https://raw.githubusercontent.com/DudaNascimento15/RegressionWorkCollege/main/fase1/datasetFase1.txt'

    print(f"Arquivo '{file_path}' . Tentando baixar do github...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Arquivo '{file_path}' baixou com sucesso.")
    else:
        print(f"Falhou baixar o arquivo {url}. Status code: {response.status_code}")
        return

    try:
        with open(file_path, "r") as f:
            for line in f:
                if line.startswith(f'x'):
                    x = line.strip()
                    x = x[x.find('[') + 1 :x.rfind(']')]
                    limitador =  ';' if x.find(';') > -1 else ','
                    x = x.split(limitador)
                    x = list(map(float, x))
                    x = np.array(x)
                if line.startswith(f'y'):
                    y = line.strip()
                    y = y[y.find('[') + 1 :y.rfind(']')]
                    limitador =  ';' if y.find(';') > -1 else ','
                    y = y.split(limitador)
                    y = list(map(float, y))
                    y = np.array(y)
                    dataset += 1
                    desenhar(x, y)
        plt.show()
    except FileNotFoundError:
        print(f"Error: O arquivo '{file_path}' não foi encontrado. Isso não deveria ocorrer.")
        print("Diretorio atual:", os.listdir('.'))

def desenhar(x, y):
    plt.figure(figsize=(4,4))
    plt.scatter(x, y)
    plt.grid(True)
    valor_correlacao = correlacao(x, y)
    b_0, b_1 = regressao(x, y)
    linha_regressao = b_0 + b_1 * x
    plt.plot(x, linha_regressao, color='red')
    plt.title(f'Dataset {dataset}. Correlação: {valor_correlacao}. β₀: {b_0}, β₁: {b_1}')

def correlacao(x, y):

    n = len(x)

    if len(x) != len(y):
        return 0

    media_x = sum(x) / n
    media_y = sum(y) / n

    numerador = sum((x[i] - media_x) * (y[i] - media_y) for i in range(n))

    denominador_x = sum((x[i] - media_x) ** 2 for i in range(n))
    denominador_y = sum((y[i] - media_y) ** 2 for i in range(n))

    denominador_total = (denominador_x * denominador_y) ** 0.5

    r = numerador / denominador_total

    return round(r, 5)

def regressao(x, y):

    n = len(x)

    if len(x) != len(y):
        return 0

    media_x = sum(x) / n
    media_y = sum(y) / n

    numerador_beta1 = sum((x[i] - media_x) * (y[i] - media_y) for i in range(n))

    denominador_beta1 = sum((x[i] - media_x) ** 2 for i in range(n))

    beta1 = numerador_beta1 / denominador_beta1
    beta0 = media_y - beta1 * media_x

    return round(beta0, 5), round(beta1, 5)

def main():
    ler_arquivo()
if __name__ == "__main__":
    main()