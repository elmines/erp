from collections import namedtuple

import tensorflow as tf

class Model(
	namedtuple("Model", ["inputs", "labels", "loss", "train_op", "predictions"])
):
	pass

#Force this class to export, because apparently it otherwise won't
#__all__ = ["Model"]
