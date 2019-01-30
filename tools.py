import numpy as np
import json
import sys

from . import data

def cleanSample(sample, maxReading, padLength):
	"""
	Clean one, individual data sample (good for inference).
	"""
	return resample(taper(sample, maxReading), padLength)

def cleanBatch(sequences, maxReading, padLength):
	"""
	Clean a batch of data samples (good for training and evaluation)
	"""
	tapered = taper(sequences, maxReading)
	return np.array([resample(sequence, padLength) for sequence in tapered])



def taper(data, maxReading):
	"""
	Taper values outside [-|maxReading|, |maxReading|]

	:param np.ndarray       data: The data to be tapered
	:param float      maxReading: The maximum (or minimum) reading

	:rtype: np.ndarray
	:return: A new, independent array with tapered values
	"""
	tapered = np.array(data, copy=True)
	maxReading = abs(maxReading)
	tapered[tapered > maxReading] = maxReading
	tapered[tapered < -maxReading] = -maxReading
	return tapered


def resample(sequence, length):
	if len(sequence) < length:
		return sequence + [sequence[-1]] * (length-len(sequence))
	else:
		return sequence[:length]

def partition(percent, inputs, labels):
	sequenceLengths = len(inputs)
	splitIndex = int(sequenceLengths * percent)
	trainSet = data.Dataset(inputs=inputs[:splitIndex], labels=labels[:splitIndex])
	testSet = data.Dataset(inputs=inputs[splitIndex:], labels=labels[splitIndex:])
	return (trainSet, testSet)

def readJson(path):
	with open(path, "r") as r:
		jsonData = json.load(r)
	return jsonData

def json2numpy(jsonData, maxPacketLength):
	"""

	:param int maxPacketLength: Maximum number of samples in a packet

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
		while j < len(data) and data[i][-1] and len(packet_set) < maxPacketLength:
			packet_set.append(data[i])
			j += 1
		if packet_set: pos.append(packet_set)

		packet_set = []
		while j < len(data) and not(data[i][-1]) and len(packet_set) < maxPacketLength:
			packet_set.append(data[i])
			j += 1
		if packet_set: neg.append(packet_set)
		i = j
	return (pos, neg)
