# By Yifei - 20054101

import time
import random
from random import shuffle
import numpy as np
import math
import matplotlib.pyplot as plt
import csv
from sklearn.neural_network import MLPClassifier
import multiprocessing as mp
import copy
import os


ann_timeout = 36000 # stop the ann after 600 minutes as the hard limit
network_configuration = [9, 9, 7]
iteration_limit = 150000 # max iterations, about 20 iterations can be iterated each second on each thread
learning_rate = 0.1 # this will be adjust by the program at runtime
alpha = 0.9
output_file_name = "outputfile.txt"


def main():
  # read data
  header, data = read_csv("GlassData.csv")

  # Normalize data, removing outliers
  data = normalize(data)
  data = removeOutliers(data)
  train_data,test_data = makeTrainTest(data)
  print("training set size =", len(train_data))
  print("test set size =", len(test_data))

  # use multi thread to train several anns, the best one will be chosen
  thread_count = mp.cpu_count() - 1 # leave user one spare thread for other tasks
  manager = mp.Manager()
  output = manager.dict()
  processes = [mp.Process(target=start_ANN_traning, args=(output,train_data,test_data,t,learning_rate,iteration_limit)) for t in range(thread_count)]
  for p in processes:
    p.start()
  for p in processes:
    p.join()

  # get the ann with the best performance
  results = [] # 0initial, 1ann, 2train set precision, 3test set precision, 4all, 5epoch
  for t in range(thread_count):
    results.append(output[t])
  results = sorted(results,key = lambda x:x[3],reverse=True) # find the one with the highest precision on the test set
  best = results[0]

  # print to console
  print_report(best)

  # output the text file
  output_files(best[0],best[1],best[5])


##
# prints out a brief report to console before the program terminates
##
def print_report(result):
  print("\nBest ann found has precision", result[3], "on test set,", result[2], "on train set after", result[5], "iterations,",
        "overall precision =",result[4])
  print("\n initial weights are")
  print("\tinput layer:")
  for neuron in result[0][0]:
    print("\t", neuron['weight'])
  print("\toutput layer:")
  for neuron in result[0][1]:
    print("\t", neuron['weight'])

  print("\n After the training, the weights became:")
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


##
# the main training cycles
##
def start_ANN_traning(output,train_data, test_data,process_number, learning_rate = 0.1,maxiteration = 5000):

  # make ann
  ann = makeNetwork(network_configuration)

  # prepare default value trackers such as error, iteration
  initial_ann = copy.deepcopy(ann)
  avg_err = 100000000
  iteration = 0
  best_ann = copy.deepcopy(ann)
  high = 0
  error_sum = 200
  start = time.time()

  # training starts
  print("start training ann number",process_number)
  while (iteration < maxiteration) and (error_sum > 30) and (time.time() - start < ann_timeout) and (avg_err > 1/3):

    iteration += 1
    enhance = []
    error_sum = 0

    # one epouch
    for row in train_data:
      actual = feedForward(ann, row[:-1])
      error = calculateError(actual, row[-1])
      coef = int((np.sum(np.square(error)) - avg_err) / 0.5)
      for i in range(coef):
        enhance.append(row)
      error_sum += np.sum(np.square(error))
      adjustWeights(error, ann, row[:-1],learning_rate)

    # retrain data the network has problem identifying the type of
    for row in enhance:
      actual = feedForward(ann, row[:-1])
      error = calculateError(actual, row[-1])
      adjustWeights(error, ann, row[:-1],learning_rate)

    # adjust learning rate
    avg_err = error_sum / len(train_data)
    learning_rate = error_sum / 200
    precision = validate(test_data, ann)

    # copy the ann with highest precision so far
    if high < precision:
      high = precision
      if error_sum < 50:
        best_ann = copy.deepcopy(ann)

  # tell the user what kind of termination happened, timeout or error floor
  if (time.time() - start) > ann_timeout:
    print("timeout, training terminated for ann number",process_number)
  else:
    print("training terminated for ann number",process_number)

  # compare the current ann (lowest error) and the best ann (highest precision)
  # the one performs better on the complete data set is chosen
  data = np.concatenate((train_data,test_data))
  if (validate(data,ann) > validate(data,best_ann)):
    best_ann = ann

  # plays a sound when training has finished for this ANN (only works on Mac)
  try:
    os.system('say "one network training has finished"')
  except:
    pass

  # record the output to multithread mamager
  output[process_number] = [initial_ann,best_ann,validate(train_data,best_ann),validate(test_data,best_ann),
                            validate(data,ann),iteration]


##
# get the percentage of correct answer the ann gives on a set of data
##
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


##
# adjust the weights using back propagation
##
def adjustWeights(error,ann,row,learning_rate = 0.1):
  old_output_layer = ann[1].copy()

  # adjust weights at the output layer
  for i in range(len(ann[1])):
    neuron = ann[1][i]
    neuron['delta'] = error[i] * neuron['output'] * (1 - neuron['output'])
    for j in range(len(neuron['weight']) - 1): # adjust weights
      adjustNeuronWeights(neuron,ann[0][j]['output'],j)
    adjustNeuronWeights(neuron,1,-1) # adjust bias

  # adjust weights at the hidden layer
  for i in range(len(ann[0])):
    neuron = ann[0][i]
    summ = 0
    for n in old_output_layer:
      summ += n['delta'] * n['weight'][i]
    neuron['delta'] = summ * neuron['output'] * (1 - neuron['output'])
    for j in range(len(neuron['weight'])-1):# adjust weights
      adjustNeuronWeights(neuron,row[j],j)
    adjustNeuronWeights(neuron,1,-1) # adjust bias


##
# adjust weights, momentum is used
##
def adjustNeuronWeights(neuron,another,j):
  last_change = neuron['change'][j]
  neuron['change'][j] = learning_rate * neuron['delta'] * another
  if ((neuron['change'][j] > 0 and last_change < 0) or (
      neuron['change'][j] < 0 and last_change > 0)):  # change direction
    if neuron['direction'][j] > 5:  # has remained in one direction for 5 times
      neuron['change'][j] += alpha * neuron['change'][j] # give it a momentum
    neuron['direction'][j] = 0
  else:  # did not change direction
    neuron['direction'][j] += 1
  neuron['weight'][j] += neuron['change'][j]


##
# calculate error = [d1,d2,d3...]
##
def calculateError(actual,expected):
  ans = [0] * 7
  ans[int(expected) - 1] = 1
  return np.subtract(ans,actual)


##
# feed data forward
##
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


##
# split data into training data and testing data
##
def makeTrainTest(data):
  np.random.shuffle(data)
  return data[:int(len(data)*4/5)],data[int(len(data)*4/5):] # train, test


def makeNetwork(net_config):
  return [[{'weight':[random.random() for n in range(net_config[0] + 1)], 'output':0, 'delta': 0, 'change':[0] * (net_config[0] + 1), "direction":[0] * (net_config[0] + 1)} for n in range(net_config[1])],
          [{'weight':[random.random() for n in range(net_config[1] + 1)], 'output':0, 'delta': 0, 'change':[0] * (net_config[1] + 1), "direction":[0] * (net_config[1] + 1)} for n in range(net_config[2])]]


##
# remove data row with two or more outliers (+- 1.5 QI)
##
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


##
# normalize data to [0,1]
##
def normalize(data):
  data_t = np.transpose(data)
  data = []
  for row in data_t[:-1]: # normalize columns that contains data
    data.append([(i-min(row))/(max(row)-min(row)) for i in row])
  data.append(data_t[-1]) # category column does not need to be normalized
  return np.transpose(data)


def output_files(initial_weights,neurons,iteration):

  output_file = open(output_file_name, "w")
  headers,output_data_set = read_csv("GlassData.csv")
  output_data = [['RI', 'Na', 'Mg', 'Al', 'Si','P', 'Ca', 'Ba', 'Fe', 'Type','Predicted']]

  output_data_set_normalized = normalize(output_data_set)


  # writing predictions
  for i in range(len(output_data_set_normalized)):
    result = feedForward(neurons, output_data_set_normalized[i][:-1])
    output_data.append(np.append(output_data_set[i],int(result.index(max(result)) + 1)))

  for row in output_data:
    line = ""
    for item in row[:-1]:
      line += str(item) + '\t'
    try:
      line += str(int(row[-1]))
    except:
      line += str(row[-1])
    output_file.write(line + '\n')
  output_file.write("\niterations used: " + str(iteration))
  output_file.write("\nterminating criteria: " + "when average error is lower than 1/3 or timeout")

  # writing the weights
  for i in range(len(neurons)): # layers
    output_file.write('\nlayer ' + str(i) + ':\n')
    for j in range(len(neurons[i])):
      output_file.write('\n\tneuron ' + str(j) + ':\n')
      for k in range(len(neurons[i][j]['weight'])):
        output_file.write('\t'+str(initial_weights[i][j]['weight'][k]) +'\t--->\t' +str(neurons[i][j]['weight'][k])+'\n')

  # writing confusion matrix
  output_data = output_data[1:]
  output_file.write('\n\n')
  confusion_matrix = []
  for type in range (1,8):
    this_type = type
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for row in output_data:
      true_type = int(row[-2])
      predicted_type = int(row[-1])
      if true_type == this_type:
        if true_type == predicted_type:
          true_positive += 1
        else:
          false_negative += 1
      else:
        if this_type != predicted_type:
          true_negative += 1
        else:
          false_positive += 1
    this_matrix = [this_type,true_positive,false_positive,true_negative,false_negative]
    if (true_positive + false_positive != 0):
      this_matrix.append((true_positive)/(true_positive+false_positive))
    else:
      this_matrix.append("NA")

    if (true_positive+false_negative != 0):
      this_matrix.append((true_positive)/(true_positive+false_negative))
    else:
      this_matrix.append("NA")
    confusion_matrix.append(this_matrix)

  for type in confusion_matrix:
    output_file.write("Type "+ str(type[0])+' Glass :\n') # type name
    output_file.write("Precision: " + str(type[-2]) + '\n')
    output_file.write("Recall: " + str(type[-1]) + '\n')
    output_file.write("T+ = " + str(type[1]) +'\t\t')
    output_file.write("F+ = " + str(type[2]) +'\t\n')
    output_file.write("T- = " + str(type[3]) +'\t')
    output_file.write("F- = " + str(type[4]) +'\t\n\n')

  output_file.close()


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