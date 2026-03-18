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
                x = x[5:-1]
                x = x.split(';')
            if line.startswith(f'y'):
                y = line.strip()
                y = y[5:-1]
                y = y.split(';')
                dataset += 1
                desenhar()
    plt.show()

def desenhar():
    plt.figure(figsize=(4,4))
    plt.scatter(x, y)
    plt.grid(True)
    plt.title(f'Dataset {dataset}')

def correlacao():
    return np.random(-1, 1)

def main():
    lerArquivo()
    desenhar()

if __name__ == "__main__":
    main()
    
    
def correlacao(x, y): 
       
    n = len(x); 
        
    if len(x) != len(y): 
        return 0;
    
    media_x = sum(x) / n;
    media_y = sum(y) / n;
    
    numerador = sum((x[i] - media_x) * (y[i] - media_y) for i in range(n));
    
    denominador_x = sum((x[i] - media_x) ** 2 for i in range(n));
    denominador_y = sum((y[i] - media_y) ** 2 for i in range(n));

    denominador_total = (denominador_x * denominador_y) ** 0.5;
     
    r = numerador / denominador_total;
    
    return round(r, 5);
    
def regressao(x, y): 
      
    n = len(x); 
        
    if len(x) != len(y): 
        return 0;
    
    media_x = sum(x) / n;    
    media_y = sum(y) / n;

    numerador_beta1 = sum((x[i] - media_x) * (y[i] - media_y) for i in range(n));
    
    denominador_beta1 = sum((x[i] - media_x) ** 2 for i in range(n));
    
    beta1 = numerador_beta1 / denominador_beta1;
    beta0 = media_y - beta1 * media_x;
    
    return round(beta0, 5), round(beta1, 5);

# TESTES MANUAIS ABAIXO
x1 = [10,8,13,9,11,14,6,4,12,7,5];
y1 = [8.04,6.95,7.58,8.81,8.33,9.96,7.24,4.26,10.84,4.82,5.68];
 

x2 = [10,8,13,9,11,14,6,4,12,7,5];
y2 = [9.14,8.14,8.47,8.77,9.26,8.10,6.13,3.10,9.13,7.26,4.74];


x3 = [8,8,8,8,8,8,8,8,8,8,19];
y3 = [6.58,5.76,7.71,8.84,8.47,7.04,5.25,5.56,7.91,6.89,12.50];

x4 = [10.0,8.0,13.0,9.0,11.0,14.0,6.0,4.0,12.0,7.0,5.0]
y4 = [7.46,6.77,12.74,7.11,7.81,8.84,6.08,5.39,8.15,6.42,5.73]

# Testando a Correlação
r = correlacao(x4, y4);
print(f"Coeficiente de Correlação (r): {r:.5f}")

# Chamando sua nova função
b0, b1 = regressao(x4, y4); 
print(f"Coeficiente beta0 (Intercepto): {b0:.5f}")
print(f"Coeficiente beta1 (Inclinação): {b1:.5f}")

