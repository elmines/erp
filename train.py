import tensorflow as tf
import numpy as np
import json

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

tf.set_random_seed(0)
np.random.seed(0)

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


trainingPercent = 0.75
paths = readJson("paths.json")	
neg = filterPackets(readJson(paths["negative"]))
pos = filterPackets(readJson(paths["positive"]))

(inputData, labelData) = prepare(neg, pos)
#Shuffle
randomIndices = np.random.permutation(len(inputData))
inputData = inputData[randomIndices]
labelData = labelData[randomIndices]

(trainingInputs, testingInputs) = partition(inputData, trainingPercent)
(trainingLabels, testingLabels) = partition(labelData, trainingPercent)

numEpochs = 20
#TRAINING
#Graph
inputs = tf.placeholder(tf.float32, (None, trainingInputs.shape[-1]) )
labels = tf.placeholder(tf.float32, None)
logit = tf.squeeze(tf.layers.dense(inputs, 1))
loss = tf.losses.sigmoid_cross_entropy(labels, logit)
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)

#TESTING
#Graph
predictions = tf.cast(tf.round(tf.nn.sigmoid(logit)), tf.int32)

numNegTraining = sum(trainingLabels == 1)
numPosTraining = len(trainingLabels) - numNegTraining
numNegTesting = sum(testingLabels == 1)
numPosTesting = len(testingLabels) - numNegTesting

print("Training on {} samples for {} epochs".format(len(trainingInputs), numEpochs))
print("\t{} negative training cases, {} positive training cases".format(numNegTraining, numPosTraining))
print("\t{} negative testing  cases, {} positive testing  cases".format(numNegTesting, numPosTesting))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	bestAccuracy = -1
	for i in range(numEpochs):
		[lossValue, _] = sess.run([loss, train_op], {inputs: trainingInputs, labels:trainingLabels})
		print("Epoch {}: loss = {}".format(i, lossValue), flush=True)

		print("Testing on {} samples".format(len(testingInputs)))
		[predictionValues] = sess.run([predictions], {inputs: testingInputs} )
		accuracy = np.mean( testingLabels == predictionValues )
		print("\t   accuracy =", accuracy)
		print()
		if accuracy >= bestAccuracy: bestAccuracy = accuracy
		else:                        break

print("LDA Model", flush=True)
ldaModel = LinearDiscriminantAnalysis()
ldaModel.fit(trainingInputs, trainingLabels)
ldaPredictions = ldaModel.predict(testingInputs)
ldaAccuracy = np.mean( testingLabels == ldaPredictions)
print("accuracy =", ldaAccuracy)

