'''
Author: Garett MacGowan
Student Number: 10197107
CISC 452 Neural and Genetic Computing
Description: This file implements a Kohonen network for two clusters
Required Libraries:
    Numpy -> pip install numpy
    sklearn -> pip install scikit-learn
    matplotlib -> pip install matplotlib
    math
'''

import numpy as np
import math
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def main(relFilePath, hasColumnLabel, clusterCount, epochs, learningRate, shouldVisualize):
  data = readData(relFilePath, hasColumnLabel)
  '''
  Translating all data points into positive direction for maxnet.
  Will translate back to original state later TODO
  '''
  data, translationApplied = translatePositive(data)
  network = initializeNetwork(data, clusterCount)
  #plotting(data, np.transpose(network['feedforwardWeights']))
  network = train(data, network, epochs, learningRate)
  # Assigning data to clusters
  clusterA, clusterB = assignCluster(data, network)
  if (shouldVisualize):
    #print('network shape ', network['feedforwardWeights'].shape)
    #plotting(data, np.transpose(network['feedforwardWeights']))
    plotting(clusterA, clusterB)

'''
Reads the data from the relative file path and returns it as a numpy array
'''
def readData(relFilePath, hasColumnLabel):
  data = np.genfromtxt(relFilePath, delimiter=',')
  # Removing the column labels
  if (hasColumnLabel):
    data = data[1:, :]
  # Randomizing the order
  np.random.shuffle(data)
  return data

'''
This function translates all datapoints into the positive number space so that maxnet
can be applied properly.
'''
def translatePositive(data):
  # need to increase all values in data by the minimum (if it is < 0)
  minimum = np.amin(data)
  if (minimum < 0):
    translation = minimum*-1
    data = np.add(translation, data)
  return data, translation

'''
This function undoes the translation from translatePositive() so that data and clusters are represented
as they were originally.
'''
def translateNegative(data, translation):
  data = np.subtract(data, translation)
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
  maximum = np.amax(data)
  # 3 rows, 2 columns for current dataset and clusterCount
  # TODO consider changing weights to be generated with each axis range in mind
  feedforwardWeights = np.random.uniform(low=maximum*0.5, high=maximum*0.8, size=(data.shape[1], clusterCount))
  # 2 rows, 1 column: each output node has an inhibitory connecion
  recurrentWeights = np.full((clusterCount, 1), recurrentWeightValue)
  return {
    'feedforwardWeights': feedforwardWeights,
    'recurrentWeights': recurrentWeights
    }

'''
Defines the training function which trains the weight vectors for the kohonen network
'''
def train(data, network, epochs, learningRate):
  # learningModifier is used to decrease learning rate over time
  learningModifier = 1
  currentTotalActivations = 0
  previousTotalActivations = 0
  for epoch in range(epochs):
    for index, row, in enumerate(data):
      winningNodeIndex, activations = feedForward(row, network)
      currentTotalActivations += np.sum(activations)
      # Don't really need to assign 'network =' here, but it is better for understanding
      network = updateWeights(network, row, winningNodeIndex, learningRate*(1/learningModifier))
    if (((epoch + 1) % (epochs/8)) == 0):
      print(str(round(epoch/epochs*100, 3)) + '% complete')
      learningModifier += 1
    # TODO determine early stopping behaviour
    #Early stop if the total activations decrease (convergence)
    print('currentTotalActivations \n ', currentTotalActivations)
    print('previousTotalActivations \n ', previousTotalActivations)
    if (currentTotalActivations < previousTotalActivations):
      break
    previousTotalActivations = currentTotalActivations
    currentTotalActivations = 0
  return network

'''
Defines the function which generates the final clustering
'''
def assignCluster(data, network):
  clusterA = np.empty((0, data.shape[1]))
  clusterB = np.empty((0, data.shape[1]))
  for index, row, in enumerate(data):
    winningNodeIndex, activations = feedForward(row, network)
    if (winningNodeIndex == 0):
      clusterA = np.append(clusterA, [row], axis=0)
    else:
      clusterB = np.append(clusterB, [row], axis=0)
  return clusterA, clusterB

'''
Defines the weight updating function
'''
def updateWeights(network, row, winningNodeIndex, learningRate):
  # Creating an array to fill with weight deltas
  deltaWeights = np.zeros(network['feedforwardWeights'].shape)
  # Subtracting the winning node weights from the input
  delta = np.subtract(row, network['feedforwardWeights'][:, winningNodeIndex])
  # Applying the learning rate
  deltaWeights[:, winningNodeIndex] = np.dot(learningRate, delta)
  # Updating the weights
  network['feedforwardWeights'] = np.add(network['feedforwardWeights'], deltaWeights)
  return network

'''
Defines the maxnet function which applies inhibitory signals to all output nodes until
a single winner is found.
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
  # Produces (2, 1) activations
  # Sum of weighted inputs for each output node. TODO reshape dynamically
  activations = np.reshape(np.dot(data, network['feedforwardWeights']), (2, 1))
  # Loop until only one activation is non-zero
  while (int(np.count_nonzero(activations, axis=0)) > 1):
    activations = maxnet(activations, network)
  # Returns the index of the winning node
  return int(np.nonzero(activations)[0]), activations

'''
Helper function for visualizing data
'''
def plotting(dataA, dataB):
  fig = plt.figure()
  ax = Axes3D(fig)
  ax.invert_xaxis()
  # plot data
  ax.plot(dataA[:,0], dataA[:,1], dataA[:,2], 'ok', c=(0,0,0,1))
  # plot dataB
  ax.plot(dataB[:,0], dataB[:,1], dataB[:,2], 'ok', c=(0,1,0,1))
  plt.show()

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
  Int: epochs
  Float: LearningRate
  Boolean: if the cluster centers should be visualized
'''
main('dataset_noclass.csv', True, 2, 2000, 0.01, True)