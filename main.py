import csv
import numpy as np
import math
from random import random
from random import shuffle
from sklearn.neural_network import MLPClassifier

layer_config = [9,8,7] # number of neurons in hidden layers and output layers

c = 0.1


def main():

  global layer_config
  header, data = read_csv("GlassData.csv")
  data = normalize(data)
  shuffle(data)

  train_data = data[int(len(data)/4):]
  test_data = data[:int(len(data)/4)]
  print(data)
  n_inputs = len(data[0]) - 1

  network = make_network()
  keep_training = True

  skitrain = []
  skians = []
  for row in data:
    skitrain.append(row[:-1])
    skians.append(row[-1])
  clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes = (8), random_state = 1,max_iter= 20000)
  clf.fit(skitrain, skians)
  answer = clf.predict(skitrain)

  precision = 0
  for i in range(len(answer)):
    print(answer[i],skians[i])
    if answer[i] == skians[i]:
      precision += 1
  print(precision/len(answer))


  while keep_training:
    for row in train_data:
      output = feed_forward(row,network)
      #print("out:",output)
      errors = calculate_error(row,output)
      #print(errors)
      backpropagate_error(errors, network,row)
    error = validate(test_data,network)
    print("precision =", (len(train_data)-error) /len(train_data))


def backpropagate_error(errors, network,input): # TODO: need to update this function name
  for k in range (len(network[1])):
    error_k = errors[k]
    # adjust weights for hidden -> output layer
    neuron = network[1][k]
    sum_of_weights = 0
    for j in range (len(network[0])):
      sum_of_weights += neuron['weights'][j]
      neuron['weights'][j] += c * error_k * neuron['delta'] * network[0][j]['output']
    neuron['weights'][-1] += c * error_k * neuron['delta']

    # adjust weights for input -> hidden layer
    delta = neuron['delta']
    for j in range (len(network[0])):
      neuron = network[0][j]
      for i in range (len(neuron['weights'])):
        neuron['weights'][i] += c * error_k * delta * input[i] * sum_of_weights
      neuron['weights'][-1] += c * error_k * delta * sum_of_weights
  return

# def adjust_weights(): # TODO: depreciated
#   return

def validate(data,network):
  err = 0
  for row in data:
    output = feed_forward(row,network)
    answer = [0] * len(output)
    answer[int(row[-1])-1] = 1
    if answer != output:
      err += 1
  return err

def calculate_error(expected,actual):
  answer = [0] * len(actual)
  answer[int(expected[-1])-1] = 1
  #print("ans:",answer)
  errors = []
  for i in range(len(actual)):
    errors.append(answer[i] - actual[i])
  return errors

def feed_forward(data,network):
  input_size = len(data)
  output = []
  for neuron in network[0]:
    for i in range (input_size-1):
      neuron['output'] += neuron['weights'][i] * data[i]
    neuron['output'] += neuron['weights'][-1]
    neuron['output'] = sigmoid(neuron['output'])
    neuron['delta'] = neuron['output'] * (1 - neuron['output']) # this is not the real delta, real delta needs to be multiplied by error

  for neuron in network[1]:
    for i in range(len(network[0])):
      neuron['output'] += neuron['weights'][i] * network[0][i]['output']
    neuron['output'] += neuron['weights'][-1]
    neuron['output'] = sigmoid(neuron['output'])
    neuron['delta'] = neuron['output'] * (1 - neuron['output']) # this is not the real delta, real delta needs to be multiplied by error
    output.append(neuron['output'])
  return threshold_out(output)

def make_network():
  network = []
  for i in range(1,len(layer_config)):
    network.append([{'weights':[random() for n in range(layer_config[i-1] + 1)],
              'output':0,'delta':0} for n in range(layer_config[i])])
  return network


def threshold_out(real_out):
  max_i = 0
  for i in range(len(real_out)):
    if real_out[i] > real_out[max_i]:
      max_i = i
  arr = [0] * len(real_out)
  arr[max_i] = 1
  return arr

def sigmoid(x):
  try:
    v = 1 / (1 + math.exp(-x))
  except:
    print(x)
    exit()
  return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
  return sigmoid(x)*(1-sigmoid(x))

def normalize(data):
  data_t = np.transpose(data)
  data = []
  for row in data_t[:-1]:
    data.append([(i-min(row))/(max(row)-min(row)) for i in row])
  data.append(data_t[-1])
  return np.transpose(data)

def read_csv(filename):
  content = []
  with open(filename) as csv_file:
    for line in csv_file:
      content.append(line[:-1].split(','))
  header = content[0]
  data = []
  for arr in content[1:]:
    print([float(i) for i in arr[1:-1]])
    data.append([float(i) for i in arr[1:]])
  return np.array(header),np.array(data)


main()