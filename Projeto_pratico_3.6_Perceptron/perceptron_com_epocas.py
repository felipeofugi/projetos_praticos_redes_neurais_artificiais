# Livro Redes Neurais Artificiais para Engenharia e Ciências Aplicadas
# fundamentos teóricos e aspectos práticos - 2ª Edição
# Projeto prático - Seção 3.6
# Nesta implementação o usuário define a quantidade de épocas para realizar o treinamento da rede.
# Pode ser usada para o caso em que os dados de treinamento não são linearmente separáveis de forma que não exista um erro.

import numpy as np
import pandas as pd

# abrir o arquivo com os dados para treinamento
excel_file_treinamento = 'Tabela_secao_3_6.xls'
dados_treinamento = pd.read_excel(excel_file_treinamento)

# separa as entradas e as saídas de dados de treinamento
entradas_treinamento = dados_treinamento.iloc[0:30, 0:3].values
saidas_treinamento = dados_treinamento.iloc[0:30, 3].values

# abre dados para teste (Tabela 3.3)
excel_file_teste = 'dados_teste.xls'
dados_teste = pd.read_excel(excel_file_teste)
entradas_teste = dados_teste.iloc[0:10, 0:3].values

# realiza o treinamento da rede
def train(inputs, outputs, learning_rate, epochs):
  
  # inicializa os pesos com valores aleatórios entre 0 e 1
  w1 = np.random.uniform(0, 1)
  w2 = np.random.uniform(0, 1)
  w3 = np.random.uniform(0, 1)
  bias = np.random.uniform(0, 1)

  # imprime os valores iniciais dos pesos
  print("valores iniciais (aleatórios): ")
  print('bias: ', bias, '\nw1: ', w1, '\nw2: ', w2, '\nw3: ', w3)

  for i in range(epochs):
    for j in range(len(inputs)):
        # aqui definimos a função de ativação. Utilizaremos a sunção sinal
        if ((inputs[j][0] * w1) + (inputs[j][1] * w2) + (inputs[j][2] * w3) + (-1) * bias) == 0:
           signal = 0
        elif ((inputs[j][0] * w1) + (inputs[j][1] * w2) + (inputs[j][2] * w3) + (-1) * bias) > 0:
           signal = 1
        elif ((inputs[j][0] * w1) + (inputs[j][1] * w2) + (inputs[j][2] * w3) + (-1) * bias) < 0:
           signal = -1
    
        # atualizando os pesos após cada iteração (regra de Hebb)
        w1 = w1 + (learning_rate * (outputs[j] - signal) * inputs[j][0])
        w2 = w2 + (learning_rate * (outputs[j] - signal) * inputs[j][1])
        w3 = w3 + (learning_rate * (outputs[j] - signal) * inputs[j][2])
        bias = bias + (learning_rate * (outputs[j] - signal) * (-1))

  # imprime os valores dos pesos obtidos após os treinamentos
  print("valores finais (após o treinamento) com ", epochs, "épocas: ")
  print('bias: ', bias, '\nw1: ', w1, '\nw2: ', w2, '\nw3: ', w3)
  return w1, w2, w3, bias

# função para realizar a predição com base nos pesos obtidos no treinamento e com os dados de teste.
def predict(pesos, dados):
   output_list = []
   w1 = pesos[0]
   w2 = pesos[1]
   w3 = pesos[2]
   bias = pesos[3]
   for j in range(len(dados)):
      if ((dados[j][0] * w1) + (dados[j][1] * w2) + (dados[j][2] * w3) + (-1) * bias) == 0:
         output = 0
      elif ((dados[j][0] * w1) + (dados[j][1] * w2) + (dados[j][2] * w3) + (-1) * bias) > 0:
         output = 1
      elif ((dados[j][0] * w1) + (dados[j][1] * w2) + (dados[j][2] * w3) + (-1) * bias) < 0:
         output = -1
      output_list.append(output)
   return output_list

# chama a função para realizar o treinamento da rede usando os dados de treinamento (Tabela 3.6) e retorna uma Tuple com os pesos obtidos
pesos = train(entradas_treinamento, saidas_treinamento, 0.01, 10000)

# realiza a classificação usando os pesos obtidos no treinamento e retorna uma lista com as saídas obtidas a partir dos dados de teste (Tabela 3.3)
output_list = predict(pesos, entradas_teste)

# imprime as saídas obtidas a partir da classificação
print("saídas obtidas pela rede treinada (usando dados de teste): \n", output_list)



