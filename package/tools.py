import numpy as np
import json
import sys

from . import data

def maxReading():
	return 80


def readJson(path):
	with open(path, "r") as r:
		jsonData = json.load(r)
	return jsonData

def resample(sequence, length, display=False):
	if len(sequence) < length:
		if display: print("Padding by {} packets".format(length-len(sequence)))
		return sequence + [sequence[-1]] * (length-len(sequence))
	else:
		if display: print("Truncating {} packets".format(len(sequence)-length))
		return sequence[:length]

def resampleBatch(sequences, length):
	return np.array([resample(sequence, length) for sequence in sequences])

def preprocess(jsonData, padLength=256):
	processed = []
	for epoch in jsonData:
		if not any(dirty(packet["data"]) for packet in epoch) and len(epoch) > 0:
			processed.append( resample(epoch, padLength) )

	arrayed = [ [packet["data"] for packet in epoch] for epoch in processed ]
	arrayed = np.array(arrayed)
	arrayed = np.transpose(arrayed, (0, 2, 1))

	return arrayed


def dirty(sequence):
	return any(sample < -maxReading() or maxReading() < sample for sample in sequence)

def taper(data):
	"""
	Tapers values outside [-tools.maxReading(), tools.maxReading()]
	"""
	data[data > maxReading()] = maxReading()
	data[data < -maxReading()] = -maxReading()

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


def partition(percent, inputs, labels):
	sequenceLengths = len(inputs)
	splitIndex = int(sequenceLengths * percent)

	trainSet = data.Dataset(inputs=inputs[:splitIndex], labels=labels[:splitIndex])
	testSet = data.Dataset(inputs=inputs[splitIndex:], labels=labels[splitIndex:])

	return (trainSet, testSet)

def json2numpy(jsonData):
	"""

	:returns: Tuple of the positive samples and negative samples
	:rtype: (np.ndarray, np.ndarray)
	"""
	data = np.array( [ packet for packet in jsonData["data"] ] )

	neg = []
	pos = []
	i = 0
	while i < len(data):
		packet_set = []
		j = i
		while j < len(data) and data[i][-1] and len(packet_set) < 256:
			packet_set.append(data[i])
			j += 1
		if packet_set: pos.append(packet_set)
		packet_set = []

		while j < len(data) and not(data[i][-1]) and len(packet_set) < 256:
			packet_set.append(data[i])
			j += 1
		if packet_set: neg.append(packet_set)
		i = j

	return (pos, neg)
