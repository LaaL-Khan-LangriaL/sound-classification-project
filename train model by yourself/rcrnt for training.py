from scipy.io import wavfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc







############# taking samples over audio data population ##########
########## function for rand_feat  e.t.c #########
########## it will read clean dir. and data from df #########



def build_rand_feat():
    x = []
    y = []
    _min, _max, = float('inf'), float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        file = np.random.choice(df[df.label==rand_class].index)
        rate, wav = wavfile.read('clean/'+file)
        label = df.at[file,'label']
        rand_index = np.random.randint(0, wav.shape[0]-config.step)
        sample = wav[rand_index:rand_index+config.step]
        x_sample = mfcc(sample, rate, 
                        numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft).T
        _min = min(np.amin(x_sample), _min)
        _max = max(np.amax(x_sample), _max)
        x.append(x_sample if config.mode == 'conv' else x_sample.T)
        y.append(classes.index(label))
    x, y = np.array(x), np.array(y)
    x = (x - _min) / (_max - _min)   


#### if we will use conv net than we need reshape ########3    
    if config.mode == 'conv':
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
        
    elif config.mode == 'time':
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2])
    y = to_categorical(y, num_classes=0)
    return x, y    
        





######## function for conv model ################
    ##### 16 filters and 3 by 3 conv #####
    #### 2d classification is used #########
    
def get_conv_model():  
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', strides=(1, 1),                     
                     padding='same', input_shape=input_shape))
                      
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1),                            
                            padding='same'))
     
     
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1),
                            padding='same'))
     
     
     
    model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1),
                            padding='same'))
    
    model.add(MaxPool2D((2, 2))) 
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'],)
    return model
        
    



##########3 function for recurrent model #####################
    ###### shape of data for RNN is (n, time, feature) ##########
##### lSTM layers are like as Dense layers but these are long short term memory unit #############
    

def get_recurrent_model():
     model = Sequential()
     model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
     model.add(LSTM(128, return_sequences=True))
     model.add(Dropout(0.5))
     model.add(TimeDistributed(Dense(64, activation='relu')))
     model.add(TimeDistributed(Dense(32, activation='relu')))
     model.add(TimeDistributed(Dense(16, activation='relu')))
     model.add(TimeDistributed(Dense(8, activation='relu')))
     model.add(Flatten())
     model.add(Dense(10, activation='softmax'))
     model.summary()
     model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'],)
     return model
     
     
     
     
     


########## write code for conv model then check first to compile it#######
########main diffrence will be shape of data ##########

class config:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=512, rate=16000):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10)
   



############## class distribution grap ##################





df = pd.read_csv('instrument.csv')
df.set_index('fname', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('clean/'+f)
    df.at[f, 'length'] = signal.shape[0]/rate

classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()








########## random choices and include this row in variable explorer it will show in spider only if u are using anaconda  ###########
####### chances for drawing  , can see in prob_dist row #############
n_samples = 2 * int(df['length'].sum()/0.1)
prob_dist = class_dist / class_dist.sum()
choices = np.random.choice(class_dist.index, p=prob_dist)


######### choices function end for ploting graph ##########



















fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
       shadow=False, startangle=90)
ax.axis('equal')
plt.show()



config = config(mode='time')   ##### this is with conv model for first chk   ######


######## build a function for fear , randome at the top ###########
if config.mode == 'conv':
     x, y = build_rand_feat()
     y_flat =np.argmax(y, axis=1)   #### map back orignal colmns ####
     input_shape = (x.shape[1], x.shape[2], 1)
     model = get_conv_model()

elif config.mode == 'time':     
     x, y = build_rand_feat()
     x, y = build_rand_feat()
     y_flat =np.argmax(y, axis=1)   
     input_shape = (x.shape[1], x.shape[2])
     model = get_recurrent_model()
   
    
    
    
    
    
    
 ##### class weight  that will get calc by sk.learn.utilits   ############


class_weight = compute_class_weight('balanced',                                      
                                     np.unique(y_flat),                                     
                                     y_flat)   
     
     
 
##### epochs are  one forward/backward    pass######
 ##### batch size training no of  examples  in one 1 backward/forward pass #######
model.fit(x, y, epochs=10, batch_size=32,
          shuffle=True,          
          class_weight=class_weight)
          









    