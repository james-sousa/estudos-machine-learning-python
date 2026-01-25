import numpy as np

class RedeNeural(object):
    def __init__(self, camadas):
        self.numeros_de_camadas = len(camadas)
        self.camadas = camadas
        self.vieses = [np.random.randn(y, 1) for y in camadas[1:]]
        self.pesos = [np.random.randn(y, x) for x, y in zip(camadas[:-1], camadas[1:])]
    
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def feedforward(self, a):
        for b, w in zip(self.vieses, self.pesos):
            a = self.sigmoid(np.dot(w, a) + b)
        return a
    
    def treinar(self, dados_treinamento, epocas,tamanho_lote, taxa_aprendizado, teste_dados=None):
        dados_treinamento = list(dados_treinamento)
        n = len(dados_treinamento)

        if teste_dados:
            teste_dados = list(teste_dados)
            n_teste = len(teste_dados)

        for epoca in range(epocas):
            np.random.shuffle(dados_treinamento)
            lotes = [dados_treinamento[k:k+tamanho_lote] for k in range(0, n, tamanho_lote)]
            for lote in lotes:
                self.atualizar_mini_lote(lote, taxa_aprendizado)
            if teste_dados:
                print(f"Época {epoca}: {self.avaliar(teste_dados)} / {len(teste_dados)}")
            else:
                print(f"Época {epoca} concluída")
    
    def atualizar_mini_lote(self, lote, taxa_aprendizado):
        nabla_b = [np.zeros(b.shape) for b in self.vieses]
        nabla_w = [np.zeros(w.shape) for w in self.pesos]

        for x, y in lote:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        self.pesos = [w - (taxa_aprendizado / len(lote)) * nw for w, nw in zip(self.pesos, nabla_w)]
        self.vieses = [b - (taxa_aprendizado / len(lote)) * nb for b, nb in zip(self.vieses, nabla_b)]
    
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.vieses]
        nabla_w = [np.zeros(w.shape) for w in self.pesos]

        # Feedforward
        ativacao = x

        #lista que armazena todas as ativações, camada por camada
        ativacoes = [x]
        #lista que armazena todos os z, camada por camada
        zs = []
        
        for b, w in zip(self.vieses, self.pesos):
            z = np.dot(w, ativacao) + b
            zs.append(z)
            ativacao = self.sigmoid(z)
            ativacoes.append(ativacao)

        # Backward pass
        delta = self.custo_derivada(ativacoes[-1], y) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, ativacoes[-2].T)

        for l in range(2, self.numeros_de_camadas):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.pesos[-l + 1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, ativacoes[-l - 1].T)
        
        return (nabla_b, nabla_w)