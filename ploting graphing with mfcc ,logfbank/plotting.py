import os
from tqdm import tqdm
import pandas as pd  ##### data frame use hoa hai ########
import numpy as np    ####fft function k liea isne computations ki hain ######
import matplotlib.pyplot as plt  #### ploting k liea use hoa hai #######
from scipy.io import wavfile   #### wavfiles read krne k liea scipy use hoa hai ######
from python_speech_features import mfcc, logfbank  ###speech signal k liea mfcc   and   filter bank energy ######
import librosa  ####ye speech k liea use hoti hai ######


###signal ke liea ploting graph ######


def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1
            
            
            
########### fft  fast fourior transform ########3

            
            
def plot_fft(fft):
   fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
   fig.suptitle('Fourier Transforms', size=16)
   i = 0
   for x in range(2):
        for y in range(5):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1
            
            
            
            
            
            
            
            
            
            
########### filter bank energy bank ploting ####################







def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Filter bank Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1




            
            
            
            
############ mel freq. capstrum  ploting#####################




def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1
            
   







################ envelope for freq ######### aik signal pass krein ge aor threshold dein ge ###########         

########33 actually mask joke audio signal k outer line show kre ga ######




def envelope(y, rate, threshold):
    mask = []        
    y = pd.Series(y).apply(np.abs)  #### numpy k zariyea Series (where Series is case sensitive ) S capital hai  ####        
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)  
        else:
            mask.append(False)  
    return mask                       #### mask k agay () ye brackets ni ayein gi ##############
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    ########### make a function to calculate fft ###############


def calc_fft(y, rate):
     n = len(y)
     freq = np.fft.rfftfreq(n, d=1.0/rate)  ###numpy is used and (d) is inverse of rate ######3
     Y = abs (np.fft.rfft(y)/n)           ### for absolute value ###
     return(Y, freq) 
    
                         
           
            
            
        
     
        ########3 reading data with pandas data frame signal rate , length etc #####
        
df = pd.read_csv('instrument.csv')
df.set_index('fname', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('wavfiles/'+f)
    df.at[f, 'length'] = signal.shape[0]/rate
    
 
    
    
    
    
    
    
    
    ####### class distribution with series of pandas #####
    
classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean() 








###### ploting figure graphically and distributing class each class contain 30 samples #################   


fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
       shadow=False, startangle=90)
ax.axis('equal')
plt.show()
df.reset_index(inplace=True)
    


signals = {}            ########Dictionary ########
fft = {}
fbank = {}
mfccs = {} 





############## Giving first file of each class total will 10 of 300       ####################

######## mask function ko is mei call kia gea hai also yaad rakhna hai ########

for c in classes:
        wav_file = df[df.label == c].iloc[0,0]
        signal, rate = librosa.load('wavfiles/'+wav_file, sr=44100)
        mask = envelope(signal, rate, 0.0005)
        signal = signal[mask]
        signals[c] = signal
        fft[c] = calc_fft(signal, rate)
        
        
        #### log f bank from speech features ######
        
        bank = logfbank(signal[:rate], rate, nfilt=26 ,nfft=1103).T           ### nfft is window size  44100/40 =1102.5 ########
        fbank[c] = bank                                                       #### store values #####
        mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T    ###signal rate is 1 second , where .T is used for transpose #######
        mfccs[c] = mel
            







########## plotting graph of signals ##########


plot_signals(signals)
plt.show()


plot_fft(fft)
plt.show()



plot_fbank(fbank)
plt.show()


plot_mfccs(mfccs)
plt.show()

        
     








   
 ############## cleaning the audio files ###########


if len(os.listdir('clean')) == 0:
    for f in tqdm (df.fname):
        signal, rate = librosa.load('wavfiles/'+f, sr=16000)
        mask = envelope (signal, rate, 0.0005)
        wavfile.write(filename='clean/'+f, rate=rate, data=signal[mask])
    



           
            
            
            