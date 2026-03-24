import matplotlib.pyplot as plt
import numpy as np

dataset = 0
x = []
y = []

def lerArquivo():
    global x, y, dataset
    with open("fase1\datasetFase1.txt", "r") as f:
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

def desenhar(x, y):
    plt.figure(figsize=(4,4))
    plt.scatter(x, y)
    #plt.xlim(left=0)
    #plt.ylim(bottom=0)
    plt.grid(True)
    valor_correlacao = correlacao(x, y)
    b_0, b_1 = regressao(x, y)
    linha_regressao = b_0 + b_1 * x
    plt.plot(x, linha_regressao, color='red')
    plt.title(f'Dataset {dataset}. Correlação: {valor_correlacao}. Regressão: {b_0}, {b_1}')
    
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
    lerArquivo()
    desenhar()

if __name__ == "__main__":
    main()


# 3)
# O dataset que não é muito apropriado pra regressão linear é o dataset 2.
# isso pq olhando o gráfico ele não segue uma reta direito,
# os pontos fazem mais uma curva do q uma linha.
# então mesmo q saia um valor de correlacao e uma reta, isso pode enganar,
# porque a relação ali não é linear de vdd.

# 4)
# no dataset 4 tem um ponto q está bem fora dos outros
# antes de ajustar a regressao, tinha q ver esse ponto melhor primeiro,
# pra saber se não é erro de medida, erro de digitação ou algum valor
# muito fora do padrão.
# isso pq um ponto assim pode puxar a reta e estragar o resultado,
# deixando a regressao meio errada ou enganosa.
