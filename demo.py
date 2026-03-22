import matplotlib.pyplot as plt
import numpy as np

dataset = 0
x = []
y = []

def lerArquivo():
    global x, y, dataset
    with open("datasetFase1.txt", "r") as f:
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