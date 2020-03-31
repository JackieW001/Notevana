"""
Convolutional Neural Net EMNIST
"""

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

from NISTparser import NIST_parser ###########################################
import matplotlib.pyplot as plt
import numpy as np
# ~~~~~~~~~~~~~~~~ Create the Convo Neural Net Model ~~~~~~~~~~~~~~~~ 
def cnn_model():
	# create model
	model = Sequential()
	#model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	#model.add(Conv2D(32, (5, 5), border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
	model.add(Conv2D(32, (5, 5), padding='valid', input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.4))
	
	'''
	#model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(Conv2D(64, (2, 2), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.4))
	'''
	model.add(Flatten())
	
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.4))
	
	model.add(Dense(50, activation='relu'))
	model.add(Dropout(0.4))
	
	model.add(Dense(62, activation='softmax'))

	
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], )
	
	return model
	
def build_model(num_samples, savefile):

	np.random.seed(7)
	X_train, X_test, y_train, y_test = NIST_parser(num_samples)

	# build the model
	print("Building model...")
	model = cnn_model()
	
	# Callback
	earlyStopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.01, patience=5, verbose=0, mode='auto')
	
	
	# fit the model
	print("Fitting model...")
	history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, callbacks=[earlyStopping])

	# Save model and datasets
	model.save(savefile)
	print( ("Model saved in %s")%(savefile) )

	# Final evaluation of the model
	scores = model.evaluate(X_test, y_test, verbose=0)
	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	
	# summarize history for accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()


	print ("Finished")

