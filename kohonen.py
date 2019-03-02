'''
Author: Garett MacGowan
Student Number: 10197107
CISC 452 Neural and Genetic Computing
Description: This file implements a Kohonen network for two clusters
Required Libraries:
    Numpy -> pip install numpy
    math
    sklearn -> pip install scikit-learn
'''

import numpy as np
import math
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

def main(relFilePath, hasColumnLabel, clusterCount):
  data = readData(relFilePath, hasColumnLabel)
  network = initializeNetwork(data, clusterCount)
  network = train(data, network, 1) # data, network, epochs
  # data = normalize(data)

# Reads the data from the relative file path and returns it as a numpy array
def readData(relFilePath, hasColumnLabel):
  data = np.genfromtxt(relFilePath, delimiter=',')
  # Removing the column labels
  if (hasColumnLabel):
    data = data[1:, :]
  return data

'''
Initializing the network:
  The cluster count represents the number of output nodes
  Each input in the input layer is fully connected with the output nodes
  Each output node will have inhibitory connections with other output nodes using a maxnet
'''
def initializeNetwork(data, clusterCount):
  # The weight value for the inhibitory conenctions
  recurrentWeightValue = 1 / clusterCount
  '''
  Finding the minimum and maximum values in the dataset so that the weights that represent
  my centroids are within an appropriate range.
  '''
  minimum = np.amin(data)
  maximum = np.amax(data)
  # 3 rows, 2 columns for current dataset and clusterCount
  feedForwardWeights = np.random.uniform(low=minimum, high=maximum, size=(data.shape[1], clusterCount))
  # 2 rows, 1 column: each output node has an inhibitory connecion
  recurrentWeights = np.full((clusterCount, 1), recurrentWeightValue)
  return {
    'feedForwardWeights': feedForwardWeights,
    'recurrentWeights': recurrentWeights
    }

'''
Defines the training function which trains the weight vectors for the kohonen network
'''
def train(data, network, epochs):
  for epoch in range(epochs):
    for index, row, in enumerate(data):
      feedForward(row, network)
      # TODO apply learning rule
      quit()

'''
Defines the maxnet function
'''
def maxnet(activations, network):
  # Creating empty numpy array to store new activations
  newActivations = np.empty(shape=activations.shape)
  for index, row, in enumerate(activations):
    # applying inhibitory connections
    temp = np.subtract(row, np.multiply(network['recurrentWeights'][index], np.subtract(np.sum(activations), row)))
    maximum = np.maximum(np.zeros((1)), temp)
    # assigning new activation
    newActivations[index] = maximum
  return newActivations

'''
Feeds forward a single data point and determines the winning cluster.
That is, the cluster for which the data point resides
'''
def feedForward(data, network):
  print('feeding forward')
  print('\n data ', data)
  print('\n network ', network['recurrentWeights'].shape)
  # Produces (2, 1) activations
  # TODO reshape dynamically
  activations = np.reshape(np.dot(data, network['feedForwardWeights']), (2, 1))
  # Loop until only one activation is non-zero
  while (int(np.count_nonzero(activations, axis=0)) > 1):
    activations = maxnet(activations, network)
  print('activations \n', activations)
  # TODO decode the winning node and return it

'''
Defines the linear activation function (sum of weighted inputs)
'''
# '''
# Normalizing all attributes to a range between 0 and 1 via max normalization.
# This makes it so that each attribute is valued equally
# '''
# def normalize(data):
#   # Don't want to normalize the class column
#   dataToNormalize = data[:, :-1]
#   classLabels = data[:, -1:]
#   normalizedData = dataToNormalize / dataToNormalize.max(axis=0)
#   normalizedData = np.concatenate((normalizedData, classLabels), axis=1)
#   return normalizedData

def outputDesignAndPerformance(initialWeights, finalWeights, learningRate, momentum, hiddenLayers, nodesPerHiddenLayer, classCount, precisionAndRecallArray, confusionMatrixArray):
  text_file = open('DesignAndPerformance.txt', 'w')
  text_file.write('Author: Garett MacGowan \n')
  text_file.write('Student Number: 10197107 \n')
  text_file.write('\n')
  text_file.write('Initial weights: \n')
  for iw in list(initialWeights):
    text_file.write(str(iw) + '\n')
  text_file.write('\n')
  text_file.write('Final weights: \n')
  for fw in list(finalWeights):
    text_file.write(str(fw) + '\n')
  text_file.write('\n')
  text_file.close()

'''
Parameters are:
  String: relative file path to data,
  Boolean: if the data has column labels,
  Int: number of clusters

'''
main('dataset_noclass.csv', True, 2)
