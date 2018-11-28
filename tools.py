import numpy as np
import json
def readJson(path):
	with open(path, "r") as r:
		jsonData = json.load(r)
	return jsonData
#Remove nulls, keep backs 
def filterPackets(jsonData):
	filtered = []
	for packet in jsonData:
		if packet is None: continue
		allChannelsClean = True 
		for channel in packet["data"]:
			if any(sample < -80 or 80 < sample for sample in channel):
				allChannelsClean = False
				break
		if allChannelsClean:
			filtered.append(packet["data"] )
	return filtered
def prepare(neg, pos):
	"""
	neg - List-like of shape [numNegPackets, numChannels, numSamples]
	pos - List-like of shape [numPosPackets, numChannels, numSamples]

	Returns:
		(inputArray, labelsArray)
		Both are np.ndarray's 
		inputArray  has shape [numNegPackets+numPosPackets, numChannels*numSamples]
		labelsArray has shape [numNegPackets+numPosPackets]
	"""
	neg = np.array(neg)
	pos = np.array(pos)
	inputArray = np.concatenate( (neg, pos), axis = 0 )
	inputArray = np.reshape(inputArray, (inputArray.shape[0], np.prod(inputArray.shape[1:])))
	labelsArray = np.array([0]*len(neg) + [1]*len(pos))
	return (inputArray, labelsArray)
def partition(sequence: list, trainingPercent: float):
	trainingIndex = int(len(sequence) * trainingPercent)
	trainingSet = sequence[:trainingIndex]
	testSet = sequence[trainingIndex:]
	return (trainingSet, testSet)

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

