from __future__ import absolute_import, division, print_function
from Lib.wer import wer
import numpy as np

from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)

print("in tune.py")
from train import final_model, train_model, AudioGenerator, int_sequence_to_text, create_json

print("in tune.py")
import zipfile
import sys
import os
import json
from IPython.display import Audio
import shutil
from Lib.char_map import char_map, index_map

def get_predictions(data_gen, index, partition, input_to_softmax, model_path):
    """ Print a model's decoded predictions
    Params:
        data_gen: AudioGenerator object
        index (int): The example you would like to visualize
        partition (str): One of 'train' or 'validation'
        input_to_softmax (Model): The acoustic model
        model_path (str): Path to saved acoustic model's weights
    """
    
    """
    # load the train and test data
    data_gen = AudioGenerator(spectrogram=False)
    
    data_gen.load_train_data(desc_file=os.path.join('Tmp','train_corpus.json'))   # necessary to calculate mean and variance
    
    data_gen.load_validation_data(desc_file=os.path.join('Tmp','valid_corpus.json'))
    """
    
    # obtain the true transcription and the audio features 
    if partition == 'validation':
        transcr = data_gen.valid_texts[index]
        audio_path = data_gen.valid_audio_paths[index]
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    elif partition == 'train':
        transcr = data_gen.train_texts[index]
        audio_path = data_gen.train_audio_paths[index]
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    elif partition == 'test':
        transcr = data_gen.test_texts[index]
        audio_path = data_gen.test_audio_paths[index]
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    else:
        raise Exception('Invalid partition!  Must be "train" or "validation" or "test"')
        
    # obtain and decode the acoustic model's predictions
    input_to_softmax.load_weights(model_path)
    prediction = input_to_softmax.predict(np.expand_dims(data_point, axis=0))
    #print("prediction.shape: " + str(prediction.shape))
    output_length = [input_to_softmax.output_length(data_point.shape[0])] 
    pred_ints = (K.eval(K.ctc_decode(
                prediction, output_length)[0][0])+1).flatten().tolist()
    #print("pred_ints: " + str(pred_ints))
    #print("len(pred_ints): " + str(len(pred_ints)))
    
    return transcr, ''.join(int_sequence_to_text(pred_ints))
    
    # play the audio file, and display the true and predicted transcriptions
    
    """
    
    print('-'*80)
    Audio(audio_path)
    print('True transcription:\n' + '\n' + transcr)
    print('-'*80)
    print('Predicted transcription:\n' + '\n' + ''.join(int_sequence_to_text(pred_ints)))
    print('-'*80)
    
    """
    
def main():
    paths = sys.argv[1:]
    """
    ['./Data/Train/Under_90_min_tuning/data.zip', './Data/Validation/Validation_10_percent/data.zip',
     './tuning_results.txt', './hyperparameter.txt']
    """
    
    with zipfile.ZipFile(paths[0],"r") as zip_ref:
        if not os.path.exists("Tmp"):
            os.mkdir("Tmp")
        if not os.path.exists(os.path.join("Tmp","train")):
            os.mkdir(os.path.join("Tmp","train"))
        zip_ref.extractall(os.path.join("Tmp","train"))  # remove it at last
         
    
    create_json(os.path.join("Tmp","train"), os.path.join('Tmp','train_corpus.json'))
           
    
    with zipfile.ZipFile(paths[1],"r") as zip_ref:
        if not os.path.exists("Tmp"):
            os.mkdir("Tmp")
        if not os.path.exists(os.path.join("Tmp","validation")):
            os.mkdir(os.path.join("Tmp","validation"))
        zip_ref.extractall(os.path.join("Tmp","validation"))  # remove it at last
        
    
    create_json(os.path.join("Tmp","validation"), os.path.join('Tmp','valid_corpus.json'))
    
    number_of_validation_examples =  sum(1 for line in open(os.path.join('Tmp','valid_corpus.json')))
    print("number of validation examples: " + str(number_of_validation_examples))
    
    with open(paths[2], 'w') as f:  # create file for writing
        pass
    
    
    filters = [150,200,250]
    kernel_size = [7,11,15]
    units = [150,200,250]
    
    # load the train and test data
    data_gen = AudioGenerator(spectrogram=False)
    
    data_gen.load_train_data(desc_file=os.path.join('Tmp','train_corpus.json'))   # necessary to calculate mean and variance
    
    data_gen.load_validation_data(desc_file=os.path.join('Tmp','valid_corpus.json'))
    
    print("validation examples in datagen: " + str(len(data_gen.valid_texts)))
    
    for f in filters:
        for k in kernel_size:
            for u in units:
                model_end = final_model(input_dim=13,
                                        filters=f,  # loaded from hyperparameter
                                        kernel_size=k,  # loaded from hyperparameter
                                        conv_stride=2,
                                        conv_border_mode='valid',
                                        units=u,  # loaded from hyperparameter
                                        activation='relu',
                                        cell=GRU,
                                        dropout_rate=1,
                                        number_of_layers=2)
                
                train_model(input_to_softmax=model_end, 
                            #pickle_path=os.path.join('Tmp','model_end.pickle'),
                            train_json=os.path.join('Tmp','train_corpus.json'),
                            valid_json=os.path.join('Tmp','valid_corpus.json'),
                            save_model_path=os.path.join('Tmp','model.h5'), 
                            epochs=10,  # changed
                            spectrogram=False)
    
                wer_sum = 0
                
                for i in range(number_of_validation_examples):
                    actual, predicted = get_predictions(data_gen, i, 'validation', model_end, 
                                                        os.path.join('Tmp','model.h5'))
                    
                    wer_sum += wer(actual.split(), predicted.split())
                
                wer_sum /= number_of_validation_examples
                
                with open(paths[2], 'a') as file:  # open file for writing
                    file.write(str(f) + " " + str(k) + " " + str(u) + " " + str(wer_sum)+"\n")
                    
                    
    
    # now take the lowest wer and have the hyperparameters to paths[3]
               
    lines = []
    with open(paths[2],'r') as f:
        lines = f.readlines()
        
    min_wer = sys.maxsize
    min_filter = None
    min_kernel_size = None
    min_units = None
        
    for l in lines:
        s = l.split()
        if float(s[3]) < min_wer:
            min_wer = float(s[3])
            min_filter = int(s[0])
            min_kernel_size = int(s[1])
            min_units = int(s[2])
            
    with open(paths[3],'w') as f:
        f.write(str(min_filter) + "\n")
        f.write(str(min_kernel_size) + "\n")
        f.write(str(min_units) + "\n")
        
                
    shutil.rmtree("Tmp")
    
    
if __name__ == "__main__":
    main()