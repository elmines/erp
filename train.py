import tensorflow as tf
tf.set_random_seed(0)
import numpy as np
import json

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

#def accuracy(predictions, labels):


paths = readJson("paths.json")	
negTrain = filterPackets(readJson(paths["training"]["negative"]))
posTrain = filterPackets(readJson(paths["training"]["positive"]))
(inputTrain, labelsTrain) = prepare(negTrain, posTrain)
numEpochs = 20

#TRAINING
#Graph
inputs = tf.placeholder(tf.float32, (None, inputTrain.shape[-1]) )
labels = tf.placeholder(tf.float32, None)
logits = tf.squeeze(tf.layers.dense(inputs, 1))
loss = tf.losses.sigmoid_cross_entropy(labels, logits)
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)

#TESTING
negTest = filterPackets(readJson(paths["testing"]["negative"]))
posTest = filterPackets(readJson(paths["testing"]["positive"]))
(inputTest, labelsTest) = prepare(negTest, posTest)
#Graph
predictions = tf.cast(tf.round(tf.nn.sigmoid(logits)), tf.int32)

print("Training on {} samples for {} epochs".format(len(inputTrain), numEpochs))
print("\t{} negative training cases, {} positive training cases".format(len(negTrain), len(posTrain)))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	bestAccuracy = -1
	for i in range(numEpochs):
		[lossValue, _] = sess.run([loss, train_op], {inputs: inputTrain, labels:labelsTrain})
		print("Epoch {}: loss = {}".format(i, lossValue))

		print("Testing on {} samples".format(len(inputTest)))
		[predictionValues] = sess.run([predictions], {inputs: inputTest} )
		accuracy = np.mean( labelsTest == predictionValues )
		print("\t   accuracy =", accuracy)
		print()
		if accuracy >= bestAccuracy: bestAccuracy = accuracy
		else:                        break

print("LDA Model")
ldaModel = LinearDiscriminantAnalysis()
ldaModel.fit(inputTrain, labelsTrain)
ldaPredictions = ldaModel.predict(inputTest)
ldaAccuracy = np.mean( labelsTest == ldaPredictions)
print("accuracy =", ldaAccuracy)

