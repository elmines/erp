import numpy as np

def accuracy(predictions, labels):
	return np.mean( predictions == labels )

def precision(predictions, labels):
	posIndices = np.where(predictions == 1)
	return np.mean(labels[posIndices])

def recall(predictions, labels):
	posIndices = np.where(labels == 1)
	return np.mean(predictions[posIndices])

def f1(predictions, labels):
	precisionVal = precision(predictions, labels)
	recallVal = recall(predictions, labels)
	return (precisionVal * recallVal) / (precisionVal + recallVal)
