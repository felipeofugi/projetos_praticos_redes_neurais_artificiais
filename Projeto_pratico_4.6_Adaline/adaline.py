# Livro Redes Neurais Artificiais para Engenharia e Ciências Aplicadas
# fundamentos teóricos e aspectos práticos - 2ª Edição
# Projeto prático - Seção 4.6

import numpy as np
import pandas as pd

# abrir o arquivo com os dados para treinamento
excel_file_treinamento = 'Tabela_Secao_4_6_RNA.xls'
dados_treinamento = pd.read_excel(excel_file_treinamento)

# separa as entradas e as saídas de dados de treinamento
entradas_treinamento = dados_treinamento.iloc[0:35, 0:4].values
saidas_treinamento = dados_treinamento.iloc[0:35, 4].values

# abre dados para teste (Tabela 3.3)
excel_file_teste = 'dados_teste.xls'
dados_teste = pd.read_excel(excel_file_teste)
entradas_teste = dados_teste.iloc[0:15, 0:4].values

# realiza o treinamento da rede
def train(inputs, outputs, learning_rate, accuracy):
  
  # inicializa os pesos com valores aleatórios entre 0 e 1
  w1 = np.random.uniform(0, 1)
  w2 = np.random.uniform(0, 1)
  w3 = np.random.uniform(0, 1)
  w4 = np.random.uniform(0, 1)
  bias = np.random.uniform(0, 1)
  epochs = 0
  erro = 10

  # imprime os valores iniciais dos pesos
  print("valores iniciais (aleatórios): ")
  print('bias: ', bias, '\nw1: ', w1, '\nw2: ', w2, '\nw3: ', w3, '\nw4: ', w4)

  while (erro > accuracy):
    eqm_anterior = eqm(w1, w2, w3, w4, bias, inputs, outputs)
    for j in range(len(inputs)):
        # aqui obtemos o valor de u
        u = (inputs[j][0] * w1) + (inputs[j][1] * w2) + (inputs[j][2] * w3) + (inputs[j][3] * w4) + (-1) * bias
    
        # atualizando os pesos após cada iteração (regra de Hebb)
        w1 = w1 + (learning_rate * (outputs[j] - u) * inputs[j][0])
        w2 = w2 + (learning_rate * (outputs[j] - u) * inputs[j][1])
        w3 = w3 + (learning_rate * (outputs[j] - u) * inputs[j][2])
        w4 = w4 + (learning_rate * (outputs[j] - u) * inputs[j][3])
        bias = bias + (learning_rate * (outputs[j] - u) * (-1))
    epochs = epochs + 1
    eqm_atual = eqm(w1, w2, w3, w4, bias, inputs, outputs)
    erro = abs(eqm_anterior - eqm_atual)

  # imprime os valores dos pesos obtidos após os treinamentos
  print("valores finais (após o treinamento) com ", epochs, "épocas: ")
  print('bias: ', bias, '\nw1: ', w1, '\nw2: ', w2, '\nw3: ', w3, '\nw4: ', w4)
  return w1, w2, w3, w4, bias

# função para realizar a predição com base nos pesos obtidos no treinamento e com os dados de teste.
def predict(pesos, dados):
   output_list = []
   w1 = pesos[0]
   w2 = pesos[1]
   w3 = pesos[2]
   w4 = pesos[3]
   bias = pesos[4]
   for j in range(len(dados)):
      if ((dados[j][0] * w1) + (dados[j][1] * w2) + (dados[j][2] * w3) + (dados[j][3] * w4) + (-1) * bias) == 0:
         output = 0
      elif ((dados[j][0] * w1) + (dados[j][1] * w2) + (dados[j][2] * w3) + (dados[j][3] * w4) + (-1) * bias) > 0:
         output = 1
      elif ((dados[j][0] * w1) + (dados[j][1] * w2) + (dados[j][2] * w3) + (dados[j][3] * w4) + (-1) * bias) < 0:
         output = -1
      output_list.append(output)
   return output_list

def eqm(w1, w2, w3, w4, bias, inputs, outputs):
   eqm = 0
   for j in range(len(inputs)):
     u = (inputs[j][0] * w1) + (inputs[j][1] * w2) + (inputs[j][2] * w3) + (inputs[j][3] * w4) + (-1) * bias
     eqm = eqm + (outputs[j] - u)**2
     eqm = eqm/len(inputs)
   return eqm


# chama a função para realizar o treinamento da rede usando os dados de treinamento (Tabela 3.6) e retorna uma Tuple com os pesos obtidos
pesos = train(entradas_treinamento, saidas_treinamento, 0.0025, 0.000001)

# realiza a classificação usando os pesos obtidos no treinamento e retorna uma lista com as saídas obtidas a partir dos dados de teste (Tabela 3.3)
output_list = predict(pesos, entradas_teste)

# imprime as saídas obtidas a partir da classificação
print("saídas obtidas pela rede treinada (usando dados de teste): \n", output_list)



