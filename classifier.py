import time
from random import random
import numpy as np

n_input = 9
n_hidden = 8
n_output = 7


def main():
  header, data = read_csv("GlassData.csv")
  data = normalize(data)
  ann = makeNetwork()
  train_data,test_data = makeTrainTest(data)


  iteration = 0
  while iteration < 500:
    for row in train_data:
      actual = feedForward(ann,row)
      error = calculateError(actual,row[-1])
      adjustWeights(error,ann,row)
    precision = validate(test_data,ann)
    print(precision)


def validate(test_data,ann):
  return


def adjustWeights(error,ann,row):
  return


def calculateError(actual,expected):
  return


def feedForward(ann,row):
  return


def makeTrainTest(data):
  return data[int(len(data)/5):],data[:int(len(data)/5)] # train, test

def makeNetwork():
  return [[{'weight':[random() for n in range(n_input + 1)], 'output':0, 'delta': 0} for n in range(n_hidden + 1)],
          [{'weight':[random() for n in range(n_hidden + 1)], 'output':0, 'delta': 0} for n in range(n_output + 1)]]

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
  header = content[0]
  data = []
  for arr in content[1:]:
    data.append([float(i) for i in arr[1:]])
  return np.array(header),np.array(data)

main()