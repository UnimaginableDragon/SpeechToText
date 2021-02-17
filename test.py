from __future__ import absolute_import, division, print_function
from Lib.wer import wer
import numpy as np

from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)

print("in test.py")
from train import final_model, train_model, AudioGenerator, int_sequence_to_text, create_json

print("in test.py")
import zipfile
import sys
import os
import json
from IPython.display import Audio
import shutil
from Lib.char_map import char_map, index_map

from docx import Document
from docx.shared import Inches

from tune import get_predictions

def main():
    paths = sys.argv[1:]
    """
    ['./Data/Test/Test_10_percent/data.zip', './model.h5']
    """
    
    with zipfile.ZipFile(paths[0],"r") as zip_ref:
        if not os.path.exists("Tmp"):
            os.mkdir("Tmp")
        if not os.path.exists(os.path.join("Tmp","test")):
            os.mkdir(os.path.join("Tmp","test"))
        zip_ref.extractall(os.path.join("Tmp","test"))  # remove it at last
         
    
    create_json(os.path.join("Tmp","test"), os.path.join('Tmp','test_corpus.json'))
           
    
    number_of_test_examples =  sum(1 for line in open(os.path.join('Tmp','test_corpus.json')))
    print("number of test examples: " + str(number_of_test_examples))
    
    lines = []
    with open("hyperparameter.txt") as f:
        lines = f.readlines()   
        
    filters = int(lines[0])
    print("filters: " + str(filters))
    kernel_size = int(lines[1])
    print("kernel_size: " + str(kernel_size))
    units = int(lines[2])
    print("units: " + str(units))
    
    # load the test data
    data_gen = AudioGenerator(spectrogram=False)
    
    data_gen.load_test_data(desc_file=os.path.join('Tmp','test_corpus.json'))
    
    print("test examples in datagen: " + str(len(data_gen.test_texts)))
    
    model_end = final_model(input_dim=13,
                            filters=filters,  # loaded from hyperparameter
                            kernel_size=kernel_size,  # loaded from hyperparameter
                            conv_stride=2,
                            conv_border_mode='valid',
                            units=units,  # loaded from hyperparameter
                            activation='relu',
                            cell=GRU,
                            dropout_rate=1,
                            number_of_layers=2)
    
    
    wer_sum = 0
                
    for i in range(number_of_test_examples):
        actual, predicted = get_predictions(data_gen, i, 'test', model_end, 
                                            paths[1])
                    
        wer_sum += wer(actual.split(), predicted.split())
                
    wer_sum /= number_of_test_examples
    
    print("word error rate on test set: " + str(wer_sum) + "%")
               
        
                
    shutil.rmtree("Tmp")
    
    
if __name__ == "__main__":
    main()


