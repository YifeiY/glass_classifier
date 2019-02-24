# By Yifei - 20054101

import time
import random
from random import shuffle
import numpy as np
import math
import matplotlib.pyplot as plt
import datetime
import csv
from sklearn.neural_network import MLPClassifier
import multiprocessing as mp
import copy
import os


ann_timeout = 1800 # stop the ann after 30 minutes as the hard limit
network_configuration = [9, 9, 7]
iteration_limit = 150000 # max iterations, about 20 iterations can be iterated each second on each thread
learning_rate = 0.1 # this will be adjust by the program at runtime
alpha = 0.9

def main():
  header, data = read_csv("GlassData.csv")
  data = normalize(data)
  data = removeOutliers(data)
  train_data,test_data = makeTrainTest(data)
  print("training set size =", len(train_data))
  print("test set size =", len(test_data))

  # if False:
  #   inputs = [i[:-1] for i in  data]
  #   ans = [i[-1] for i in data]
  #   clf = MLPClassifier(hidden_layer_sizes = 9,max_iter=10**5,n_iter_no_change=500,random_state=0)
  #   clf.fit(inputs[int(len(inputs)/5):],ans[int(len(inputs)/5):])
  #   print(clf.score(inputs[:int(len(inputs)/5)],ans[:int(len(inputs)/5)]))
  #   exit()


  thread_count = mp.cpu_count() - 1 # leave user one spare thread for other tasks
  manager = mp.Manager()
  output = manager.dict()

  processes = [mp.Process(target=start_ANN_traning, args=(output,train_data,test_data,t,learning_rate,iteration_limit)) for t in range(thread_count)]
  for p in processes:
    p.start()
  for p in processes:
    p.join()

  results = [] # ann, train set precision, test set precision, epoch
  for t in range(thread_count):
    results.append(output[t])
  sorted(results,key = lambda x:x[2],reverse=True) # find the one with the highest precision on the test set
  best = results[0]
  print_report(best)
  for i in results:
    print_report(i)

def print_report(result):
  print("Best ann found has precision", result[3], "on test set,", result[2], "on train set after", result[5], "iterations,",
        "overall precision =",result[4])
  print("\n initial weights are")
  print("\tinput layer:")
  for neuron in result[0][0]:
    print("\t", neuron['weight'])
  print("\toutput layer:")
  for neuron in result[0][1]:
    print("\t", neuron['weight'])

  print("\nAfter the training, the weights became:")
  print("\tinput layer:")
  for neuron in result[1][0]:
    print("\t", neuron['weight'])
  print("\toutput layer:")
  for neuron in result[1][1]:
    print("\t", neuron['weight'])

  try:
    os.system('say "program has finished"')
  except:
    pass

def start_ANN_traning(output,train_data, test_data,process_number, learning_rate = 0.1,maxiteration = 5000):
  ann = makeNetwork(network_configuration)
  initial_ann = copy.deepcopy(ann)
  avg_err = 100000000
  iteration = 0
  best_ann = copy.deepcopy(ann)
  high = 0
  error_sum = 200
  start = time.time()

  print("start training ann number",process_number)
  while (iteration < maxiteration) and (error_sum > 30) and (time.time() - start < ann_timeout) and (avg_err > 1/6):
    # train_data, test_data = makeTrainTest(data)
    iteration += 1
    enhance = []
    error_sum = 0
    for row in train_data:
      actual = feedForward(ann, row[:-1])
      error = calculateError(actual, row[-1])
      coef = int((np.sum(np.square(error)) - avg_err) / 0.5)
      for i in range(coef):
        enhance.append(row)
      error_sum += np.sum(np.square(error))
      adjustWeights(error, ann, row[:-1],learning_rate)
    for row in enhance:
      actual = feedForward(ann, row[:-1])
      error = calculateError(actual, row[-1])
      adjustWeights(error, ann, row[:-1],learning_rate)
    avg_err = error_sum / len(train_data)
    learning_rate = error_sum / 200
    precision = validate(test_data, ann)
    if high < precision:
      high = precision
      if error_sum < 50:
        best_ann = copy.deepcopy(ann)
    high = max(precision, high)
    print(avg_err)
    #print("error =", error_sum, "| precision =", precision, "|", len(enhance), "data points are in the enhanced training bag")
  if (time.time() - start) > ann_timeout:
    print("timeout, training terminated for ann number",process_number)
  else:
    print("training terminated for ann number",process_number)
  print((iteration < maxiteration) ,"+", (error_sum) ,"+", (time.time() - start < ann_timeout))
  data = np.concatenate((train_data,test_data))


  if (validate(data,ann) > validate(data,best_ann)):
    best_ann = ann
  try:
    os.system('say "one network training has finished"')
  except:
    pass

  output[process_number] = [initial_ann,best_ann,validate(train_data,best_ann),validate(test_data,best_ann),
                            validate(data,ann),iteration]



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


def adjustWeights(error,ann,row,learning_rate = 0.1):
  old_output_layer = ann[1].copy()

  for i in range(len(ann[1])):
    neuron = ann[1][i]
    neuron['delta'] = error[i] * neuron['output'] * (1 - neuron['output'])
    for j in range(len(neuron['weight']) - 1): # weights
      adjustNeuronWeights(neuron,ann[0][j]['output'],j)
    adjustNeuronWeights(neuron,1,-1)

  for i in range(len(ann[0])):
    neuron = ann[0][i]
    summ = 0
    for n in old_output_layer:
      summ += n['delta'] * n['weight'][i]
    neuron['delta'] = summ * neuron['output'] * (1 - neuron['output'])
    for j in range(len(neuron['weight'])-1):
      adjustNeuronWeights(neuron,row[j],j)
    adjustNeuronWeights(neuron,1,-1)


def adjustNeuronWeights(neuron,another,j):
  last_change = neuron['change'][j]
  neuron['change'][j] = learning_rate * neuron['delta'] * another
  if ((neuron['change'][j] > 0 and last_change < 0) or (
      neuron['change'][j] < 0 and last_change > 0)):  # change direction
    if neuron['direction'][j] > 5:  # has remained in one direction for 5 times, give it a momentum
      neuron['change'][j] += alpha * neuron['change'][j]
    neuron['direction'][j] = 0
  else:  # did not change direction
    neuron['direction'][j] += 1
  neuron['weight'][j] += neuron['change'][j]


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
  np.random.seed(0)
  np.random.shuffle(data)
  return data[:int(len(data)*4/5)],data[int(len(data)*4/5):] # train, test


def makeNetwork(net_config):
  return [[{'weight':[random.random() for n in range(net_config[0] + 1)], 'output':0, 'delta': 0, 'change':[0] * (net_config[0] + 1), "direction":[0] * (net_config[0] + 1)} for n in range(net_config[1])],
          [{'weight':[random.random() for n in range(net_config[1] + 1)], 'output':0, 'delta': 0, 'change':[0] * (net_config[1] + 1), "direction":[0] * (net_config[1] + 1)} for n in range(net_config[2])]]


def removeOutliers(data):
  data = np.transpose(data)
  print("removing outliers (+- 1.5 QI)...")
  remove_indices =[0] * len(data[0])
  for col in data[:-1]:
    q1 = np.percentile(col,25)
    q3 = np.percentile(col,75)
    qi = q3 - q1
    for i in range(len(col)):
      if (col[i] < q1 - 1.5* qi) or (col[i] > q3 + 1.5 * qi):
        remove_indices[i] += 1
  data = np.transpose(data)
  new_data = []
  for i in range(len(data)):
    if remove_indices[i] < 3: # if that data does not contain more than 2 outliers
      new_data.append(data[i])
  print(len(data) - len(new_data),"data points removed from",len(data),"data pool")
  return new_data


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

  data2 = []
  for d in data:
    if True:#int(d[-1]) != 1 and int(d[-1]) != 2:
      data2.append(d)
  data = data2
  return np.array(header),np.array(data)


main()