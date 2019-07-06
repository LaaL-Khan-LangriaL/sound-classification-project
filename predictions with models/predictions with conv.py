import os
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model
from sklearn.metrics import accuracy_score







##########prediction ka function##########

##### softmax used for individual class probability ############

def build_predictions(audio_dir):
    y_true = []
    y_pred = []
    fn_prob = {}
    
    
    
    print('Extracting Features From Audio')
    for fn in tqdm(os.listdir(audio_dir)):
        rate, wav = wavfile.read(os.path.join(audio_dir, fn))
        label = fn2class[fn]
        c = classes.index(label)
        y_prob = []
        
        
        for i in range(0, wav.shape[0]-config.step, config.step):
            sample = wav[i:i+config.step]
            x = mfcc(sample, rate, numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft)
            x = (x - config.min) / (config.max - config.min) 
            
            
            if config.mode == 'conv':   ### for conv gray scale channel ####
                
                ##### conv k liea [1] pehlay aye ga ####
               x = x.reshape(1, x.shape[1], x.shape[0], 1)
            elif config.mode == 'time':
                x = np.expand_dims(x, axis=0)
                
                
            y_hat = model.predict(x)
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat))
            y_true.append(c)    
         
            
            
            
            
            #### file name for storing result in y prob list for predictions ###
                
            
        fn_prob[fn] = np.mean(y_prob, axis=0).flatten() 
        
    return y_true, y_pred, fn_prob
    
      
      
       
    
    
df = pd.read_csv('instrument.csv')
classes = list(np.unique(df.label)) 
fn2class = dict(zip(df.fname, df.label))
p_path = os.path.join('pickles', 'conv.p')


with open(p_path, 'rb') as handle:
    config = pickle.load(handle)

model = load_model(config.model_path)

y_true, y_pred, fn_prob = build_predictions('clean')
acc_score = accuracy_score(y_true=y_true, y_pred=y_pred) 



### will fill the data frame for every 10 sec to pred all catagories #########
y_probs = []
for i, row in df.iterrows():
     y_prob = fn_prob[row.fname]
     y_probs.append(y_prob)
     for c, p in zip(classes, y_prob):   ####prob is 1/10 coz of 10 classes###
         df.at[i, c] = p   ### inplace operation for zip together and for call class###
                            ##### p is the associate prob for class ####


y_pred = [classes[np.argmax(y)] for y in y_probs]
df['y_pred'] = y_pred    #### new coloum for pred in data frame  ###



##### output in csv file in data frame ####
df.to_csv('predictions with conv.csv', index=False)       

                            
 