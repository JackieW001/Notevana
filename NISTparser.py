import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from img_read2 import resize_and_normalize_NIST_image
from scipy.misc import imread
import numpy as np

# shuffle input_dataset and mlb_tags
def shuffle_in_unison(a, b):
	rng_state = np.random.get_state()
	np.random.shuffle(a)
	np.random.set_state(rng_state)
	np.random.shuffle(b)
	return a,b
		
def NIST_parser(num_samples):
	nist_imgs = []
	nist_tags = []

	# get NIST data
	try:
		# nist folder from this script
		nist_path = "./NIST/"
		NIST = os.listdir(nist_path)
	except IOError:
		print 'Remember to unzip NIST.zip. Please keep NIST.zip but delete NIST folder when done.'
	
	# reading through NIST folder: extracting .png images and corresponding hexcode tags
	for super_cat in NIST: #caps, digits, small folders
		if (super_cat != '.DS_Store'):
			print super_cat
			new_path = os.path.join(nist_path, super_cat)
			super_cat = os.listdir(new_path)
			#super_index = 0
			
			for cat in super_cat: # A-Z, a-z, 0-9
				if (cat != '.DS_Store'):
					print("---> " + cat)
					new_sub_path = os.path.join(new_path, cat)
					cat = os.listdir(new_sub_path)
					index = 0;
		
					for image in cat:
						if index < num_samples:
							# getting the first ten images from each category
							read_image = imread(os.path.join(new_sub_path,image))
							nist_imgs.append(read_image)
							# getting the hexcode
							hexcode = image.split('_')[1]
							hexcode = hexcode.decode('hex')
							nist_tags.append(hexcode)
							index = index + 1 
	'''					
	# resize_and_normalize_image(<list of images>)
	# input_dataset will be split into train and validate set and then shuffle it
	input_dataset = resize_and_normalize_NIST_image(nist_imgs)
	#print(nist_tags)

	# one hot encoded nist tags
	mlb = MultiLabelBinarizer()
	mlb_tags = mlb.fit_transform(s.split(', ') for s in nist_tags)

	
	input_dataset, mlb_tags = shuffle_in_unison(input_dataset, mlb_tags)
	mnist = []
	mnist.append(input_dataset)
	mnist.append(mlb_tags)
	#mnist = np.array(mnist)
	# print("input dataset")
	# print(mnist[0])
	# print("mlb_tags")
	# print(mnist[1])

	#split data into train and test data
	X_train, X_test, y_train, y_test = train_test_split(input_dataset, mlb_tags, train_size = 0.8)
	return X_train, X_test, y_train, y_test
	'''
	# one hot encoded nist tags
	mlb = MultiLabelBinarizer()
	mlb_tags = mlb.fit_transform(s.split(', ') for s in nist_tags)
	
	nist_imgs, mlb_tags = shuffle_in_unison(nist_imgs, mlb_tags)
	
	X_train, X_test, y_train, y_test = train_test_split(nist_imgs, mlb_tags, train_size = 0.8)
	
	
	X_train = resize_and_normalize_NIST_image(X_train)
	X_test = resize_and_normalize_NIST_image(X_test)
	
	return X_train, X_test, y_train, y_test

# For testing
NIST_parser(10)
"""
X_train = train[0][0]
y_train = train[0][1]
X_test = test[0][0]
y_test = test[0][1]

print(train)
#print(y_train)
"""