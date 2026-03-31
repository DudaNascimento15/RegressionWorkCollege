import pandas as pd
import requests
import matplotlib.pyplot as plt


def baixar_arquivo_git(caminho_arquivo):
    url = f"https://raw.githubusercontent.com/DudaNascimento15/RegressionWorkCollege/main/fase3/{caminho_arquivo}"

    print(f"Arquivo '{caminho_arquivo}' . Tentando baixar do github...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(caminho_arquivo, "wb") as f:
            f.write(response.content)
        print(f"Arquivo '{caminho_arquivo}' baixou com sucesso.")
    else:
        print(f"Falhou baixar o arquivo {url}. Status code: {response.status_code}")
        return


def ler_arquivos_csv(caminho_arquivo):
    data = pd.read_csv(caminho_arquivo, header=None)
    return data


def carregar_arquivo(nome_arquivo):
    baixar_arquivo_git(nome_arquivo)
    return ler_arquivos_csv(nome_arquivo)


def grafico_dispersao(x, y):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Dados originais', zorder=5)

    plt.title('Gráfico de Dispersão - Dados data_preg')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def main():
    arquivo_usuario = input("Digite o nome do arquivo: ").strip()

    df = carregar_arquivo(arquivo_usuario)
    if df is None:
        return

    grafico_dispersao(df[0].values, df[1].values)

def calcular_residuo(y_observado, y_previsto):
    return pow(y_observado - y_previsto, 2)

def calcular_eqm(y_observado, y_previsto):
    somatorio = 0
    n = y_observado.size()

    for i in range(n):
        somatorio += calcular_residuo(y_observado[i], y_previsto[i])

    resultado = (1/n) * somatorio

    return resultado


if __name__ == "__main__":
    main()
