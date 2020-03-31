from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import NISTparser
from keras import backend as K
K.set_image_dim_ordering('th')

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
	
# ~~~~~~~~~~~~~~~~ Create the Convo Neural Net Model ~~~~~~~~~~~~~~~~ 
def cnn_model():
	# create model
	model = Sequential()
	#model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	
	#model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(Conv2D(64, (2, 2), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	
	model.add(Dense(3, activation='softmax'))
	
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
	
def build_model(num_samples):
	X_train, X_test, y_train, y_test = NISTparser.NIST_parser(num_samples)
	print X_train.shape, X_test.shape, y_train.shape, y_test.shape
	
	# build the model
	print("Building model...")
	model = cnn_model()
	#class_weight = {'0' : 33., 'a': 33., 'A': 33.}
	
	"""
	# Instantiate the label encoder
	le = LabelEncoder()
	
	class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
	sample_weights = compute_sample_weight('balanced', y_train)

	class_weight = dict(zip(le.transform(list(le.classes_)), class_weights))
	"""
	# fit the model
	print("Fitting model...")
	model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=128)
	
	# Save model and datasets
	model.save('character_recognition_cnn_mock.h5')
	print("Model saved in character_recognition_cnn.h5")

	# Final evaluation of the model
	scores = model.evaluate(X_test, y_test, verbose=0)
	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
