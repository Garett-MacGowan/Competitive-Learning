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

def main(relFilePath, hasColumnLabel):
  data = readData(relFilePath, hasColumnLabel)
  # data = normalize(data)

# Reads the data from the relative file path and stores it as a numpy array
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
  Each output node will have inhibitory connections with other output nodes
'''
def initializeNetwork(data, clusterCount):
  minimum = np.amin(data)
  maximum = np.amax(data)

  weights = np.random.rand(data.shape[1], clusterCount)
  weights = np.random.uniform(low=-1.0, high=1.0, size=(hiddenLayerNodes, attributeCount))


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

'''
main('GlassData.csv', True)
