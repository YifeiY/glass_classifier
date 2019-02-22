import time
import random
from random import shuffle
import numpy as np
import math
import matplotlib.pyplot as plt
import datetime
import csv

n_input = 9
n_hidden = 20
n_output = 7
learning_rate = 0.1

def main():
  global learning_rate
  header, data = read_csv("GlassData.csv")
  data = normalize(data)
  ann = makeNetwork()
  train_data,test_data = makeTrainTest(data)


  visual_arr = []
  avg_err = 100000000
  iteration = 0
  high = 0
  while iteration < 500000000:
    #train_data, test_data = makeTrainTest(data)
    iteration+=1
    enhance = []
    error_sum = 0
    for row in train_data:
      actual = feedForward(ann,row[:-1])
      error = calculateError(actual,row[-1])
      coef = int((np.sum(np.square(error)) - avg_err)/0.5)
      for i in range (coef):
        enhance.append(row)
      error_sum += np.sum(np.square(error))
      adjustWeights(error,ann,row[:-1])
    for row in enhance:
      actual = feedForward(ann, row[:-1])
      error = calculateError(actual, row[-1])
      adjustWeights(error, ann, row[:-1])
    avg_err = error_sum/len(train_data)
    learning_rate = error_sum/300
    precision = validate(test_data,ann)
    high = max(precision,high)
    print("error =",error_sum,"precision =", precision,"high =",high,len(enhance))


def validate(test_data,ann):

  correct = 0
  for row in test_data:
    actual = feedForward(ann, row[:-1])
    maxv = 0
    maxi = 0
    for i in range(len(actual)):
      if actual[i] > maxv:
        maxv = actual[i]
        maxi = i
    if maxi == int(row[-1]) - 1:
      correct += 1
  return correct/len(test_data)


def adjustWeights(error,ann,row):
  old_output_layer = ann[1].copy()

  for i in range(len(ann[1])):
    neuron = ann[1][i]
    neuron['delta'] = error[i] * neuron['output'] * (1 - neuron['output'])
    for j in range(len(neuron['weight']) - 1): # weights
      neuron['weight'][j] += learning_rate * neuron['delta'] * ann[0][j]['output']
    neuron['weight'][-1] += learning_rate * neuron['delta'] * 1 # bias


  for i in range(len(ann[0])):
    neuron = ann[0][i]
    summ = 0
    for n in old_output_layer:
      summ += n['delta'] * n['weight'][i]
    neuron['delta'] = summ * neuron['output'] * (1 - neuron['output'])
    for j in range(len(neuron['weight'])-1):
      neuron['weight'][j] += learning_rate * neuron['delta'] * row[j]
    neuron['weight'][-1] += learning_rate * neuron['delta']


def calculateError(actual,expected):
  ans = [0] * 7
  ans[int(expected) - 1] = 1
  return np.subtract(ans,actual)


def feedForward(ann,inputs):
  output = []
  for neuron in ann[0]:
    neuron['output'] = sigmoid(np.dot(np.append(inputs,[1]),neuron['weight']))
  for neuron in ann[1]:
    neuron['output'] = sigmoid(np.dot(np.append([n['output'] for n in ann[0]],[1]),neuron['weight']))
    output.append(neuron['output'])
  return output






def sigmoid(x):
  if x > 500:
    return 1
  if x < -500:
    return 0
  return 1 / (1 + math.exp(-x))

def makeTrainTest(data):

  np.random.shuffle(data)

  # with open(str(time.time())+'.csv', mode='w') as file:
  #   filewriter = csv.writer(file)
  #   for r in data:
  #     filewriter.writerow(r)
  #   file.close()

  return data[:int(len(data)*4/5)],data[int(len(data)*4/5):] # train, test

def makeNetwork():
  return [[{'weight':[random.random() for n in range(n_input + 1)], 'output':0, 'delta': 0} for n in range(n_hidden)],
          [{'weight':[random.random() for n in range(n_hidden + 1)], 'output':0, 'delta': 0} for n in range(n_output)]]

def normalize(data):
  data_t = np.transpose(data)
  data = []
  for row in data_t[:-1]: # normalize columns that contains data
    data.append([(i-min(row))/(max(row)-min(row)) for i in row])
  data.append(data_t[-1]) # category column does not need to be normalized
  return np.transpose(data)

def read_csv(filename):
  content = []
  with open(filename) as csv_file:
    for line in csv_file:
      content.append(line[:-1].split(','))
    csv_file.close()
  header = content[0]
  data = []
  for arr in content[1:]:
    data.append([float(i) for i in arr[1:]])
  return np.array(header),np.array(data)

main()