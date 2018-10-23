import tensorflow as tf
tf.set_random_seed(0)
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


paths = readJson("paths.json")	
negTrain = filterPackets(readJson(paths["training"]["negative"]))
posTrain = filterPackets(readJson(paths["training"]["positive"]))
(inputTrain, labelsTrain) = prepare(negTrain, posTrain)
numEpochs = 20

#Graph
inputs = tf.placeholder(tf.float32, (None, inputTrain.shape[-1]) )
labels = tf.placeholder(tf.float32, None)
logits = tf.squeeze(tf.layers.dense(inputs, 1))
loss = tf.losses.sigmoid_cross_entropy(labels, logits)
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)

#Training
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(numEpochs):
		[lossValue, _] = sess.run([loss, train_op], {inputs: inputTrain, labels:labelsTrain})
		print("Epoch {}: loss = {}".format(i, lossValue))

