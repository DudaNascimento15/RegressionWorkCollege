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
def ajustar_modelo(x, y, grau):
    return np.polyfit(x, y, grau)

def calcular_polinomio_manual(x, coefs):
    grau = len(coefs) - 1
    y_pred = np.zeros_like(x, dtype=float)
    for i, coef in enumerate(coefs):
        exp = grau - i
        y_pred += coef * (x ** exp)
    return y_pred

def grafico_dispersao_com_regressoes(x, y):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color="blue", label="Dados originais", zorder=5)

    ind = np.argsort(x)
    x_ord = x[ind]

    # grau 1
    coef_1 = ajustar_modelo(x, y, 1)
    y_1 = calcular_polinomio_manual(x_ord, coef_1)
    plt.plot(x_ord, y_1, 'r', label='grau 1')
    
    # grau 2
    coef_2 = ajustar_modelo(x, y, 2)
    y_2 = calcular_polinomio_manual(x_ord, coef_2)
    plt.plot(x_ord, y_2, 'g', label='grau 2')
    
    # grau 3
    coef_3 = ajustar_modelo(x, y, 3)
    y_3 = calcular_polinomio_manual(x_ord, coef_3)
    plt.plot(x_ord, y_3, 'k', label='grau 3')

    # grau 4
    coef_4 = ajustar_modelo(x, y, 8)
    y_4 = calcular_polinomio_manual(x_ord, coef_4)
    plt.plot(x_ord, y_4, 'y', label='grau 8')

    plt.title("Gráfico de dispersão com regressão polinomial")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print("== COEFICIENTES ==")
    print(f"Grau 1: {coef_1}")
    print(f"Grau 2: {coef_2}")
    print(f"Grau 3: {coef_3}")
    print(f"Grau 4: {coef_4}")

def main():
    arquivo_usuario = input("Digite o nome do arquivo: ").strip()

    df = carregar_arquivo(arquivo_usuario)
    if df is None:
        return

    grafico_dispersao(df[0].values, df[1].values)
    grafico_dispersao_com_regressoes(df[0].values, df[1].values)
    
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
