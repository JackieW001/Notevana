import numpy as np
import os
from keras.models import load_model

def predict_results (abs_path, save_filename, cnn_model="character_recognition_cnn_v2.h5"):
	''' Loads numpy array specified by 
	   	Input: abs_path to numpy array to be loaded
	   		   save_filename to save predictions in data folder
	   	Output:saved numpy and text files of predictions in data folder
	   	---> ? always save predictions in predictions.txt and predictions.npy
	'''
	print(cnn_model)
	path = "./data/"
	save_filename = save_filename.split('.')[0]
	save_filename = os.path.join(path, save_filename)
	print(save_filename)

	# load data for testing
	try:
		X_test = np.load(abs_path)
	except OSError, e:
			sys.exit()

	# load model
	print("Loading model......................")
	model_path = cnn_model
	model = load_model(model_path)

	chararr = [ '0','1','2','3','4','5','6','7','8','9',
				'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
				'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z' ]
	# ~~~~~~~~~~~~~~~~ Predictions ~~~~~~~~~~~~~~~~
	# calculate predictions
	predictions = model.predict_classes(X_test)
	char = []
	for i in np.nditer(predictions):
		print(chararr[i])
		char.append(chararr[i])
	print("\n")
	
	# outputs decimal code of character
	print(predictions)
	
	# save predictions
	with open(save_filename + '.txt', 'w') as outfile:
		for i in char:
			outfile.write(i)
		
	print('Done')

## For testing	
#predict_results("./data/test_digit_pics.npy", "predictions")