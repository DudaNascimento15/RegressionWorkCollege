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