'''
Author: Garett MacGowan
Student Number: 10197107
CISC 452 Neural and Genetic Computing
Description:
  This file implements a simple competitive network for clustering using maxnet
  and dot product as a similarity measure. Note that a Euclidean feedforward
  function is present but not used. It can be swapped out for testing purposes.
  Execution parameters are listed at the bottom of this file. I have implemented
  a matplotlib visualization function so that the resulting clusters can be seen.
Required Libraries:
    Numpy -> pip install numpy
    sklearn -> pip install scikit-learn
    matplotlib -> pip install matplotlib
    math
    random
'''

import numpy as np
import math
import random
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
# The import below is for extra testing purposes
from sklearn.datasets.samples_generator import make_blobs

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main(relFilePath, hasColumnLabel, clusterCount, epochs, learningRate, shouldVisualize):
  data = readData(relFilePath, hasColumnLabel)
  # The commented out code below is for an alternate dataset for testing
  # data, y = make_blobs(n_samples=1000, centers=2, n_features=3, random_state=2)
  # Subtracting means so I can take use dot product as my similarity metric.
  data, columnMeans = subtractMeans(data)
  network = initializeNetwork(data, clusterCount)
  initialCentroids = np.copy(network['feedforwardWeights'])
  network = train(data, network, epochs, learningRate)
  # Adding means back into data
  data, network, initialCentroids = addMeans(data, network, initialCentroids, columnMeans)
  # Assigning data to clusters
  clusters = assignCluster(data, network)
  if (shouldVisualize):
    plotting(clusters, np.transpose(network['feedforwardWeights']), np.transpose(initialCentroids))

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
This function centers the data about the origin. It is necessary for dot product similarity
'''
def subtractMeans(data):
  columnMeans = np.mean(data, axis=0)
  data = np.subtract(data, columnMeans)
  return data, columnMeans

'''
This function adds the means back into the data
'''
def addMeans(data, network, initialCentroids, columnMeans):
  data = np.add(data, columnMeans)
  meansToAdd = np.full(network['feedforwardWeights'].shape, np.reshape(columnMeans, (columnMeans.shape[0],1)))
  network['feedforwardWeights'] = np.add(network['feedforwardWeights'], meansToAdd)
  initialCentroids = np.add(initialCentroids, meansToAdd)
  return data, network, initialCentroids

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
  minimim = np.amin(data)
  # 3 rows, 2 columns for current dataset and clusterCount
  feedforwardWeights = np.random.uniform(low=minimim*0.5, high=maximum*0.5, size=(data.shape[1], clusterCount))
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
  previousIterTotalDelta = math.inf
  for epoch in range(epochs):
    # totalDelta is used to detect convergence
    totalDelta = 0
    for index, row, in enumerate(data):
      winningNodeIndex, activations = feedForward_dotProd(row, network)
      # currentTotalActivations += np.sum(activations)
      network, deltaWeights = updateWeights(network, row, winningNodeIndex, learningRate*(1/learningModifier))
      # Check if the delta is passed the threshold
      totalDelta += abs(np.sum(deltaWeights))
    # Early stopping behaviour (when the weights stop changing the division rounds to 1)
    if (previousIterTotalDelta / totalDelta == 1):
      print('stopped early, the centroids stopped moving!')
      break
    previousIterTotalDelta = totalDelta
    if (((epoch + 1) % (epochs/10)) == 0):
      print(str(round(epoch/epochs*100, 3)) + '% complete')
      learningModifier += 1
  return network

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
  return network, deltaWeights

'''
Defines the function which generates the final clustering
'''
def assignCluster(data, network):
  clusters = []
  # Create a cluster for every output neuron
  for _ in range(network['feedforwardWeights'].shape[1]):
    clusters.append(np.empty((0, data.shape[1])))
  for index, row, in enumerate(data):
    winningNodeIndex, activations = feedForward_dotProd(row, network)
    if (winningNodeIndex == None):
      continue
    clusters[winningNodeIndex] = np.append(clusters[winningNodeIndex], [row], axis=0)
  return clusters

'''
Defines the maxnet function which applies inhibitory signals to all output nodes until
a single winner is found.
'''
def maxnet(activations, network):
  # Creating empty numpy array to store new activations
  newActivations = np.empty(shape=activations.shape)
  for index, row, in enumerate(activations):
    # Applying inhibitory connections
    temp = np.subtract(row, np.multiply(network['recurrentWeights'][index], np.subtract(np.sum(activations), row)))
    maximum = np.maximum(np.zeros((1)), temp)
    # Assigning new activation
    newActivations[index] = maximum
  return newActivations

'''
Feeds forward a single data point and determines the winning cluster.
That is, the cluster for which the data point resides
'''
def feedForward_dotProd(data, network):
  # Produces (2, 1) activations
  activations = np.reshape(np.dot(data, network['feedforwardWeights']), (network['feedforwardWeights'].shape[1], 1))
  # Loop until only one activation is non-zero
  while (int(np.count_nonzero(activations, axis=0)) > 1):
    activations = maxnet(activations, network)
  # Returns the index of the winning node
  nonzeros = np.nonzero(activations)[0]
  # Tiebreaking
  if (nonzeros.shape[0] == 0):
    nonzero = random.randint(0, network['feedforwardWeights'].shape[1]-1)
  else:
    nonzero = int(nonzeros)
  return nonzero, activations

'''
This function is not currently in use.
It is here solely for testing purposes
'''
def feedForward_euclidean(data, network):
  # Creating empty activations to be filled with Euclidean distance
  activations = np.empty((0, network['feedforwardWeights'].shape[1]))
  # Calculating squared Euclidean distance between vectors
  for index in range(0, network['feedforwardWeights'].shape[1]):
    # Distance between two vectors is the length of the difference vectors
    differenceVector = np.subtract(data, network['feedforwardWeights'][:,index])
    distance = np.dot(differenceVector, differenceVector)
    activations = np.append(activations, distance)
  while (int(np.count_nonzero(activations, axis=0)) > 1):
    activations = maxnet(activations, network)
  # Returns the index of the winning node
  nonzeros = np.nonzero(activations)[0]
  # Tiebreaking
  if (nonzeros.shape[0] == 0):
    nonzero = random.randint(0, network['feedforwardWeights'].shape[1]-1)
  else:
    nonzero = int(nonzeros)
  return nonzero, activations
  

'''
Helper function for visualizing data. Initial centroids are black circles
and final centroids are black stars. Clusters are randomly coloured.
'''
def plotting(data, centroids, initialCentroids):
  fig = plt.figure()
  ax = Axes3D(fig)
  for index in range(len(data)):
    i = index + 1
    if (data[index].shape[0] > 0):
      ax.plot(data[index][:,0], data[index][:,1], data[index][:,2], 'o', c=(random.uniform(0, 1),random.uniform(0, 1),random.uniform(0, 1),1))
  # Initial centroids are black circles
  if (initialCentroids.shape[0] > 0):
    ax.plot(initialCentroids[:,0], initialCentroids[:,1], initialCentroids[:,2], 'o', c=(0,0,0,1))
  # Centroids are black stars
  if (centroids.shape[0] > 0):
    ax.plot(centroids[:,0], centroids[:,1], centroids[:,2], '*', c=(0,0,0,1))
  plt.show()

def outputDesignAndPerformance(initialWeights, finalWeights, learningRate,):
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

def outputClustering(clusters):
  text_file = open('clusters.txt', 'w')
  text_file.write('Author: Garett MacGowan \n')
  text_file.write('Student Number: 10197107 \n')
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
main('dataset_noclass.csv', True, 2, 1000, 0.01, True)