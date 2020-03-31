from img_read2 import img_read
from predict_results import predict_results
from NISTparser import NIST_parser
from build_model import build_model


'''
build_model(500, "character_recognition_cnn_v2.h5")
'''

filename = "test_pics"
# read image
img_read( ("./%s/")%(filename), ("%s.npy")%(filename) )

#predict results
predict_results( ("./data/%s.npy")%(filename), "predictions.txt", "character_recognition_cnn_v3.h5")
