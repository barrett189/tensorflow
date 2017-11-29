import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.data import Dataset, Iterator
import numpy as np

# Nezwork architecture
n_input = 65
n_hidden1 = 60
n_hidden2 = 60
n_hidden3 = 60
n_hidden4 = 60
n_hidden5 = 30
n_hidden6 = 30
n_hidden7 = 30
n_output = 1

# Training params
n_epochs = 1000
batch_size = 500
learning_rate = 0.001



# Definition of Dataset Class
class DataSet:
	def __init__(self, batch_size):
		x = []
		y = []

		filename = 'input1.npy'
		file = open(filename, 'rb')
		data = np.load(file)
		file.close()
		x.extend(data[:, :-1])
		y.extend(data[:, -1])

		self.x_d = np.array(x)
		self.y_d = np.array(y)

		self.data = tf.contrib.data.Dataset.from_tensor_slices((tf.constant(self.x_d), tf.constant(self.y_d)))
		self.data = self.data.shuffle(buffer_size=10000)
		self.data = self.data.batch(batch_size)
		self.iterator = self.data.make_initializable_iterator()
		self.next_batch = self.iterator.get_next()


# Dataset Class initialization
tr_set = DataSet(batch_size)

# placeholders
x = tf.placeholder(tf.float32, shape=(None, n_input), name = "x")
y = tf.placeholder(tf.float32, shape=(None), name = "y")
is_training = tf.placeholder(tf.bool, shape=(), name="is_training")

bn_params = {
	'is_training':is_training,
	'decay':0.99,
	'updates_collections':None
}


# leaky rectified linear unit
def leaky_relu(z):
	return tf.maximum(0.01 * z, z)


# Network Model
def neural_network_model(data):
	
	he_init = tf.contrib.layers.variance_scaling_initializer()
	
	# Config 1
	hidden1 = fully_connected(x, n_hidden1, activation_fn=leaky_relu, weights_initializer=he_init, normalizer_fn = batch_norm, normalizer_params=bn_params)
	hidden2 = fully_connected(hidden1, n_hidden2, activation_fn=leaky_relu, weights_initializer=he_init, normalizer_fn = batch_norm, normalizer_params=bn_params)
	hidden3 = fully_connected(hidden2, n_hidden3, activation_fn=leaky_relu, weights_initializer=he_init, normalizer_fn = batch_norm, normalizer_params=bn_params)
	hidden4 = fully_connected(hidden3, n_hidden4, activation_fn=leaky_relu, weights_initializer=he_init, normalizer_fn = batch_norm, normalizer_params=bn_params)
	hidden5 = fully_connected(hidden4, n_hidden5, activation_fn=leaky_relu, weights_initializer=he_init, normalizer_fn = batch_norm, normalizer_params=bn_params)
	
	output = fully_connected(hidden5, n_output, activation_fn = tf.tanh, weights_initializer=he_init)
	output = tf.tanh(output)
	# Config 1 end

	# Config 2
	#output = fully_connected(x[25:26], n_output, activation_fn = tf.tanh, weights_initializer=he_init)
	#output = tf.tanh(output)
	# Config 2 end

	return output


# Training
def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(prediction * y)
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(-cost)
	

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(1, n_epochs):
			tr_epoch_loss = 0
			
			sess.run(tr_set.iterator.initializer)
			
			while True:
				try:
					batch_x, batch_y = sess.run(tr_set.next_batch)
					_, c = sess.run([optimizer, cost], feed_dict={is_training: True, x:batch_x, y: batch_y})
					tr_epoch_loss += c
				except tf.errors.OutOfRangeError:
					break
			print('Epoch ', epoch, 'TrainLoss: ', tr_epoch_loss)


train_neural_network(x)