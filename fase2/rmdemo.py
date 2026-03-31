import pandas as pd
import scipy.io as scipy
import os
import numpy as np
import requests
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def baixar_arquivo_git(caminho_arquivo):
    url = f"https://raw.githubusercontent.com/DudaNascimento15/RegressionWorkCollege/main/fase2/{caminho_arquivo}"

    print(f"Arquivo '{caminho_arquivo}' . Tentando baixar do github...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(caminho_arquivo, "wb") as f:
            f.write(response.content)
        print(f"Arquivo '{caminho_arquivo}' baixou com sucesso.")
    else:
        print(f"Falhou baixar o arquivo {url}. Status code: {response.status_code}")
        return


def ler_arquivos_dat(caminho_arquivo):
    mat = scipy.loadmat(caminho_arquivo)
    data = mat["data"]
    df = pd.DataFrame(data, columns=["tamanho", "numero", "preco"])
    return df


def ler_arquivos_csv(caminho_arquivo):
    df = pd.read_csv(caminho_arquivo, header=None, names=["tamanho", "numero", "preco"])
    return df


def carregar_arquivo(nome_arquivo):
    baixar_arquivo_git(nome_arquivo)
    nome, extensao = os.path.splitext(nome_arquivo)

    if extensao.lower() == ".mat":
        return ler_arquivos_dat(nome_arquivo)
    elif extensao.lower() == ".csv":
        return ler_arquivos_csv(nome_arquivo)
    else:
        print("Arquivo não suportado")
        return None


def analise_estatistica(df):
    print("Análise Estatística")
    print(df.describe())

    media = df["preco"].mean()
    print(f"\nMédia de preço das casas: {media}")

    menor_casa = df.sort_values(by="tamanho").iloc[0]
    print(f"\nMenor casa:")
    print(menor_casa)
    print(f"Quanto custa: {menor_casa['preco']}")

    casa_mais_cara = df.loc[df["preco"].idxmax()]
    print(f"\nCasa mais cara:")
    print(casa_mais_cara)
    print(f"Casa mais cara custa: {casa_mais_cara['preco']}")


def regressao_manual(df):
    """
    Regressão linear múltipla usando a equação normal:
    theta = (X^T X)^-1 X^T y
    """
    X = df[["tamanho", "numero"]].values
    y = df["preco"].values.reshape(-1, 1)

    # adiciona coluna de 1 para o intercepto
    X_b = np.c_[np.ones((X.shape[0], 1)), X]

    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return theta


def prever_manual(theta, tamanho, numero_quartos):
    x_novo = np.array([[1, tamanho, numero_quartos]])
    previsao = x_novo @ theta
    return float(previsao[0][0])


def gerarMatrizes(df):
    matriz = df[["tamanho", "numero"]].to_numpy()
    matriz_uns = np.ones((matriz.shape[0], 1))
    matriz_x = np.hstack((matriz_uns, matriz))

    variaveis_dependentes = df["preco"].to_numpy()

    return matriz_x, variaveis_dependentes


def correlacaoSimples(x, y):
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


def regressaoSimples(x, y):
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


def desenharGraficoSimples(x, y, titulo):
    plt.figure(figsize=(4,4))
    plt.scatter(x, y)
    plt.grid(True)
    valor_correlacao = correlacaoSimples(x, y)
    b_0, b_1 = regressaoSimples(x, y)
    linha_regressao = b_0 + b_1 * x
    plt.plot(x, linha_regressao, color='red')
    plt.title(f'{titulo} - Correlação: {valor_correlacao}. Regressão: {b_0}, {b_1}')


def desenharParteD(df):
    tamanho = df["tamanho"]
    quartos = df["numero"]
    preco = df["preco"]

    print("\n")
    desenharGraficoSimples(tamanho, preco, "Gráfico Tamanho x Preço")
    print("\n")    
    desenharGraficoSimples(quartos, preco, "Gráfico Quartos x Preço")
    print("\n")  


def desenharGrafico3d(df):
    #modelo = regressao_sklearn(df)
    theta = regressao_manual(df)
    correlacao_tamanho_preco = correlacaoSimples(df["tamanho"], df["preco"])
    correlacao_quartos_preco = correlacaoSimples(df["numero"], df["preco"])

    theta0 = theta [0][0] #modelo.intercept_
    theta1, theta2 = theta[1][0], theta[2][0] #modelo.coef_

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(df["tamanho"], df["numero"], df["preco"], color="blue")

    x_surf = np.linspace(df["tamanho"].min(), df["tamanho"].max(), 20)
    y_surf = np.linspace(df["numero"].min(), df["numero"].max(), 20)
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)
    z_surf = theta0 + theta1 * x_surf + theta2 * y_surf

    ax.plot_surface(
        x_surf, y_surf, z_surf, color="red", alpha=0.3, label="Plano de Regressão"
    )

    ax.set_xlabel("Tamanho (pés²)")
    ax.set_ylabel("Número de quartos")
    ax.set_zlabel("Preço")
    ax.set_title("Regressão Linear Múltipla - Preço de Casas")

    text_str = (f'Correlação (Tamanho x Preço): {correlacao_tamanho_preco:.4f}\n'
                f'Correlação (Quartos x Preço): {correlacao_quartos_preco:.4f}')
    plt.figtext(0.15, 0.8, text_str, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()
    print("\n")  


def regressao_sklearn(df):
    X = df[["tamanho", "numero"]]
    y = df["preco"]

    modelo = LinearRegression()
    modelo.fit(X, y)

    return modelo


def comparar_resultados(df):
    print("\n" + "=" * 60)
    print("REGRESSÃO LINEAR MÚLTIPLA - IMPLEMENTAÇÃO MANUAL")
    print("=" * 60)

    theta = regressao_manual(df)

    print(f"Intercepto (theta0): {theta[0][0]}")
    print(f"Coeficiente tamanho (theta1): {theta[1][0]}")
    print(f"Coeficiente número de quartos (theta2): {theta[2][0]}")

    preco_1650_3_manual = prever_manual(theta, 1650, 3)
    print(
        f"\nPreço previsto para casa de 1650 pés² e 3 quartos (manual): {preco_1650_3_manual:.0f}"
    )

    print("\nVariação da quantidade de quartos:")
    for quartos in range(1, 6):
        preco = prever_manual(theta, 1650, quartos)
        print(f"1650 pés² e {quartos} quartos -> preço previsto: {preco:.2f}")

    print("\n" + "=" * 60)
    print("COMPARAÇÃO COM SCIKIT-LEARN")
    print("=" * 60)

    modelo = regressao_sklearn(df)

    print(f"Intercepto sklearn: {modelo.intercept_}")
    print(f"Coeficientes sklearn: {modelo.coef_}")

    preco_1650_3_sklearn = modelo.predict([[1650, 3]])[0]
    print(
        f"\nPreço previsto para casa de 1650 pés² e 3 quartos (sklearn): {preco_1650_3_sklearn:.0f}"
    )

    print("\nComparação manual x sklearn:")
    print(f"Manual   : {preco_1650_3_manual:.6f}")
    print(f"Sklearn  : {preco_1650_3_sklearn:.6f}")
    print(f"Diferença: {abs(preco_1650_3_manual - preco_1650_3_sklearn):.10f}")

    return theta, modelo


def explicar_resultado(theta):
    coef_quartos = theta[2][0]

    print("\n" + "=" * 60)
    print("EXPLICAÇÃO DO ITEM (h)")
    print("=" * 60)

    print(
        "Ao aumentar ou diminuir a quantidade de quartos, o preço previsto muda linearmente."
    )
    print(
        "Isso acontece porque a regressão linear múltipla aprende um peso para cada variável."
    )

    print(f"\nCoeficiente da variável 'numero' (quartos): {coef_quartos:.6f}")

    if coef_quartos > 0:
        print(
            "Como o coeficiente é positivo, aumentar o número de quartos aumenta o preço previsto."
        )
    elif coef_quartos < 0:
        print(
            "Como o coeficiente é negativo, aumentar o número de quartos diminui o preço previsto."
        )
    else:
        print(
            "Como o coeficiente é zero, mudar o número de quartos não altera o preço previsto."
        )

    print("\nMotivo:")
    print("O modelo calcula o preço usando algo do tipo:")
    print("preco = theta0 + theta1 * tamanho + theta2 * numero")
    print("Então, ao mudar apenas 'numero', o preço varia de acordo com theta2.")


def main():
    arquivo_usuario = input(
        "Digite o nome do arquivo que você subiu (ex: data.csv ou data.mat): "
    ).strip()

    df = carregar_arquivo(arquivo_usuario)
    if df is None:
        return

    analise_estatistica(df)
    desenharParteD(df)
    desenharGrafico3d(df)
    theta, modelo = comparar_resultados(df)
    explicar_resultado(theta)


if __name__ == "__main__":
    main()
