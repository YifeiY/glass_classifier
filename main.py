import csv
import numpy as np
import math
from random import random
from random import shuffle

layer_config = [9,8,7] # number of neurons in hidden layers and output layers


def main():

  global layer_config
  header, data = read_csv("GlassData.csv")
  data = normalize(data)
  shuffle(data)

  train_data = data[int(len(data)/4):]
  test_data = data[:int(len(data)/4)]

  n_inputs = len(data[0]) - 1

  network = make_network()
  keep_training = True

  while keep_training:
    for row in train_data:
      feed_forward(row,network)
      calculate_error(row,network)
      backpropogate_error()
      adjust_weights()
      validate()


def backpropogate_error():
  return

def adjust_weights():
  return

def validate():
  return

def calculate_error(data,network):
  ans = [0] * 7
  ans[data[-1]] = 1
  output = []
  for i in range(len(ans)):
    output.append(ans[i] - network[-1][i]['output'])
  return output

def feed_forward(data,network):
  input_size = len(data)
  for neuron in network[0]:
    for i in range (input_size-1):
      neuron['output'] += neuron['weights'][i] * data[i]
    neuron['output'] += neuron['weights'][-1]
    neuron['output'] = sigmoid(neuron['output'])

  for neuron in network[1]:
    for i in range(len(network[0])):
      neuron['output'] += neuron['weights'][i] * network[0][i]['output']
    neuron['output'] += neuron['weights'][-1]
    neuron['output'] = sigmoid(neuron['output'])
    print(neuron['output'])


def make_network():
  network = []
  for i in range(1,len(layer_config)):
    network.append([{'weights':[random() for n in range(layer_config[i-1] + 1)],
              'output':0,'delta':0} for n in range(layer_config[i])])
  return network




def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
  return sigmoid(x)*(1-sigmoid(x))

def normalize(data):
  data_t = np.transpose(data)
  data = []
  for row in data_t:
    data.append([(i-min(row))/(max(row)-min(row)) for i in row])
  return np.transpose(data)

def read_csv(filename):
  content = []
  with open(filename) as csv_file:
    for line in csv_file:
      content.append(line[:-2].split(','))
  header = content[0]
  data = []
  for arr in content[1:]:
    data.append([float(i) for i in arr[1:]])
  return np.array(header),np.array(data)


main()