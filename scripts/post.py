import sys
import argparse
import json
#3rd Party Libraries
import numpy as np
import matplotlib.pyplot as plt


path = "train.json"
with open(path, "r") as r:
	log = json.load(r)

classes = log["header"]["classes"]["names"]
data = log["data"]
numRecords = len(data)

models = ["modelA", "modelB"]
numModels = len(models)


#SAMPLE_WISE
loss = np.zeros( (numRecords, numModels) )
predictions = np.zeros( (numRecords, numModels, len(classes)) )
labels     = np.zeros( (numRecords, len(classes)) )

for (i, record) in enumerate(data):
	for (j, modelName) in enumerate(models):
		labels[i] = np.array(record["labels"])
		predictions[i][j] = record[modelName]["predictions"]
		loss[i][j] = record[modelName]["loss"]


#CUMULATIVE
loss = np.cumsum(loss, axis=0) / np.expand_dims( np.arange(1, len(loss)+1) , axis=-1)
positives = np.cumsum(labels, axis=0)
predPositives = np.cumsum(predictions, axis=0)

truePositives = np.cumsum(
			np.logical_and(
				np.expand_dims(np.equal(labels, 1), axis=1) ,
				np.equal(predictions, 1) ,
			),
			axis=0
)

accuracy = np.cumsum( np.expand_dims(labels, axis=1) == predictions)
recall = truePositives / np.expand_dims(positives, axis=-1)
precision = truePositives / predPositives

recall[np.isnan(recall)] = 0.
precision[np.isnan(precision)] = 0.


modelLabels = ["eegOnly", "eegAndContext"]
def plotMetric(values, title):
	for (i, model) in enumerate(modelLabels):
		for (j, className) in enumerate(classes):
			label = f"{model}:{className}"
			plt.plot(values[:,i,j], label=label)
	plt.title(title)
	plt.xlabel("Samples Observed")
	plt.legend()
	plt.show()

plotMetric(recall, "Recall")
plotMetric(precision, "Precision")
