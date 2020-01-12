import tensorflow as tf

tf.compat.v1.enable_eager_execution()
tf.executing_eagerly()

from tensorflow.compat.v1.keras import layers
import  tensorflow.compat.v1.keras.backend as K
from tensorflow.compat.v1.keras.models import save_model
import pandas
import numpy
import numba 
import os 
import gc 
import sys 
import librosa 
import scipy 
import pywt 
import sklearn
import h5py
from tqdm import tqdm 
from scipy import signal 
from scipy import stats 
from scipy import fftpack
from statsmodels.robust import mad
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

print(tf.version.VERSION)
print(tf.keras.__version__)
print(K.floatx())
print("Finished Importing Files")
window = 150000
n_feat = 17
n_fold = 3
seed = 65
print("Finished initializing variables")
data = pandas.read_csv("train.csv",header=None,dtype=numpy.float64,float_precision='high')
print("Finished Reading the CSV File")
data = data.to_numpy()
output = data[:,1]
training = numpy.float64(data[:,0])
del data
print("Finished splitting the data into training and output")
length = len(training)
print(length)
retain_value = length-(length%window)
training = training[:retain_value:]
output = output[:retain_value:]
length = len(training)
print(length)
q = int(length/window)
print("Dropping the remainder")
assert (length%window)==0
assert (len(training)) == (len(output))
output = output[window-1::window]
training = training.reshape(-1,window)
print(training.shape)
'''
gaussian_noise = numpy.random.normal(0,0.5,window)
data = numpy.zeros([q,n_feat])
for i in tqdm(range(0,q)):
    a = training[i,:]
    a = a + gaussian_noise
    a = a - numpy.median(a)
    coeff = pywt.wavedec(a,wavelet='db4',mode='per',level=1)
    threshold = 10 
    coeff[1:] = (pywt.threshold(j,value=threshold,mode='hard') for j in coeff[1:])
    a = pywt.waverec(coeff,wavelet='db4',mode='per')
    m = librosa.feature.mfcc(a,n_mfcc=20,dct_type=2,norm='ortho',htk=True)
    m = m.mean(axis=1)
    data[i,0] = m[3]
    data[i,1] = m[17]
    data[i,2] = stats.moment(a,1)
    data[i,3] = stats.moment(a,2)
    data[i,4] = stats.moment(a,3)
    data[i,5] = numpy.std(a)
    data[i,6] = numpy.dot(a,a)
    data[i,7] = ((a[:1]*a[:-1])<0).sum()
    b = librosa.feature.rms(a,frame_length=window,hop_length=window,center=True,pad_mode='reflect')
    data[i,8:10] = b.reshape(-1)
    b = librosa.feature.spectral_contrast(a)
    data[i,10:] = b.mean(axis=1)
'''
print("Training data has been prepared , moving on to preparing the testing data")
sample_submission = pandas.read_csv("sample_submission.csv",header='infer')
PATH = "test/"
submission_len = len(sample_submission)
sample_submission = sample_submission.to_numpy()

submission = numpy.empty([submission_len,window])
#del training
for i in tqdm(range(0,submission_len)):
    temp = pandas.read_csv(PATH+sample_submission[i,0]+".csv",header='infer',dtype=numpy.int16)
    temp=temp.to_numpy()
    submission[i,:]=temp.reshape(-1)
    '''
    temp = temp + gaussian_noise.reshape(-1,1)
    temp = temp - temp.median()
    temp = temp.to_numpy(dtype=numpy.float4)
    temp = temp.reshape(-1)
    coeff = pywt.wavedec(temp,wavelet='db4',mode='per',level=1)
    sigma = mad(coeff[-1])
    threshold = 10
    coeff[1:] = (pywt.threshold(j,value=threshold,mode='hard') for j in coeff[1:])
    temp = pywt.waverec(coeff,wavelet='db4',mode='per')
    m = librosa.feature.mfcc(temp,n_mfcc=20,dct_type=2,norm='ortho',htk=True)
    m = m.mean(axis=1)
    submission[i,0] = m[3]
    submission[i,1] = m[17]
    submission[i,2] = stats.moment(temp,1)
    submission[i,3] = stats.moment(temp,2)
    submission[i,4] = stats.moment(temp,3)
    submission[i,5] = numpy.std(temp)
    submission[i,6] = numpy.dot(temp,temp)
    submission[i,7] = ((temp[:1]*temp[:-1])<0).sum()
    b = librosa.feature.rms(temp,frame_length=window,hop_length=window,center=True,pad_mode='reflect')
    submission[i,8:10] = b.reshape(-1)
    b = librosa.feature.spectral_contrast(temp)
    submission[i,10:] = b.mean(axis=1)

data = data.reshape(-1,n_feat,1)
submission = submission.reshape(-1,n_feat,1)
'''
prediction = numpy.zeros([submission_len,1])
prediction1 = numpy.zeros([submission_len,1])
prediction2 = numpy.zeros([submission_len,1])
val_output = output[1715:]
output = output[:1714]
training = training.reshape(-1,window,1)
val_train = training[1715:,:,:]
training = training[:1714,:,:]
'''
val_train = data[1715:,:].reshape(-1,n_,1)
training = data[:1714,:].reshape(-1,n_feat,1)
'''
batch = 64
dataset = tf.data.Dataset.from_tensor_slices((training,(output,output)))#,output
dataset = dataset.repeat(count=None).batch(batch_size=batch,drop_remainder=False)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
print(dataset)

val_dataset = tf.data.Dataset.from_tensor_slices((val_train,(val_output,val_output)))#,val_output
val_dataset = val_dataset.repeat(count=None).batch(batch_size=batch,drop_remainder=True)
val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

submission = tf.data.Dataset.from_tensor_slices((submission))
submission = submission.batch(batch_size=batch,drop_remainder=False)
submission = submission.prefetch(tf.data.experimental.AUTOTUNE)
del training, output , numpy , os,librosa #, a ,m
del  val_train,val_output

Model_input=tf.compat.v1.keras.Input(shape=(window,1),batch_size=batch,name="Input_Layer")

a = tf.compat.v1.keras.layers.CuDNNGRU(units=512,kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer=tf.keras.regularizers.l2(0.01),return_sequences=True,stateful=True,name="GRU0")(Model_input)
b = tf.compat.v1.keras.layers.CuDNNGRU(units=8,kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer=tf.keras.regularizers.l2(0.01),return_sequences=True,stateful=True,name="GRU1")(a)
c = tf.compat.v1.keras.layers.CuDNNGRU(units=8,kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer=tf.keras.regularizers.l2(0.01),return_sequences=True,stateful=True,name="GRU2")(b)
d = tf.compat.v1.keras.layers.CuDNNGRU(units=4,kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer=tf.keras.regularizers.l2(0.01),return_sequences=True,stateful=True,name="GRU3")(c)
e = tf.compat.v1.keras.layers.CuDNNGRU(units=4,kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer=tf.keras.regularizers.l2(0.01),return_sequences=True,stateful=True,name="GRU4")(d)
f = tf.compat.v1.keras.layers.CuDNNGRU(units=4,kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer=tf.keras.regularizers.l2(0.01),return_sequences=False,stateful=True,name="GRU5")(e)
g = tf.compat.v1.keras.layers.Dense(1,activation='relu')(f)

a0=(tf.compat.v1.keras.layers.Conv1D(filters=1, kernel_size=1, strides=1, padding='valid', data_format="channels_last", dilation_rate=1, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(0.01),name='Layer2'))(Model_input)
x0=tf.compat.v1.keras.layers.Activation('tanh')(a0)
b0=(tf.compat.v1.keras.layers.CuDNNGRU(units=1, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(0.01), return_sequences=True, stateful=False,name="Layer3"))(x0)
c0=(tf.compat.v1.keras.layers.CuDNNGRU(units=1, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(0.01),return_sequences=True, stateful=False,name="Layer4"))(b0)
#batch_1= tf.compat.v1.keras.layers.BatchNormalization()(c0)
d0=(tf.compat.v1.keras.layers.Conv1D(filters=2,kernel_size=2, strides=1, padding='valid', data_format="channels_last", dilation_rate=1, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(0.01),name="Layer5"))(c0)#batch_1
x1=tf.compat.v1.keras.layers.Activation('tanh')(d0)
e0=(tf.compat.v1.keras.layers.CuDNNGRU(units=2, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(0.01), return_sequences=True, stateful=False,name="Layer6"))(x1)
f0=(tf.compat.v1.keras.layers.CuDNNGRU(units=2, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(0.01), return_sequences=True, stateful=False,name="Layer7"))(e0)
#batch_2= tf.compat.v1.keras.layers.BatchNormalization()(f0)
g0=(tf.compat.v1.keras.layers.Conv1D(filters=2, kernel_size=1, strides=1, padding='valid', data_format="channels_last", dilation_rate=1, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(0.01),name="Layer11"))(f0)#batch_2
x2=tf.compat.v1.keras.layers.Activation('tanh')(g0)
h0=(tf.compat.v1.keras.layers.CuDNNGRU(units=2, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(0.01), return_sequences=True, stateful=False,name="Layer12"))(x2)
i0=(tf.compat.v1.keras.layers.CuDNNGRU(units=2, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(0.01), return_sequences=True, stateful=False,name="Layer13"))(h0)
#batch_3= tf.compat.v1.keras.layers.BatchNormalization()(i0)
j0=(tf.compat.v1.keras.layers.Flatten(data_format="channels_last",name="Layer14"))(i0)#batch_3
k0=(tf.compat.v1.keras.layers.Dense(units=1,activation='tanh',kernel_initializer='glorot_uniform',kernel_regularizer=tf.keras.regularizers.l2(0.01),name="Layer15"))(j0)
l0=(tf.compat.v1.keras.layers.Dense(units=1,activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=tf.keras.regularizers.l2(0.01),name="OLayer"))(k0)


a1=(tf.compat.v1.keras.layers.CuDNNGRU(units=1, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(0.01), return_sequences=True, stateful=False,name="Layer_2"))(Model_input)
b1=(tf.compat.v1.keras.layers.CuDNNGRU(units=1, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(0.01),return_sequences=True, stateful=False,name="Layer_3"))(a1)
c1=(tf.compat.v1.keras.layers.Conv1D(filters=1, kernel_size=1, strides=1, padding='valid', data_format="channels_last", dilation_rate=1, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(0.01),name='Layer_4'))(b1)
batch_11= tf.compat.v1.keras.layers.BatchNormalization()(c1)
y0=tf.compat.v1.keras.layers.Activation('tanh')(batch_11)
d1=(tf.compat.v1.keras.layers.CuDNNGRU(units=2, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(0.01), return_sequences=True, stateful=False,name="Layer_5"))(y0)
e1=(tf.compat.v1.keras.layers.CuDNNGRU(units=2, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(0.01), return_sequences=True, stateful=False,name="Layer_6"))(d1)
f1=(tf.compat.v1.keras.layers.Conv1D(filters=2,kernel_size=1, strides=1, padding='valid', data_format="channels_last", dilation_rate=1, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(0.01),name="Layer_7"))(e1)
batch_21 = tf.compat.v1.keras.layers.BatchNormalization()(f1)
y1=tf.compat.v1.keras.layers.Activation('tanh')(batch_21)
g1=(tf.compat.v1.keras.layers.CuDNNGRU(units=2, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(0.01), return_sequences=True, stateful=False,name="Layer_8"))(y1)
h1=(tf.compat.v1.keras.layers.CuDNNGRU(units=2, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(0.01), return_sequences=True, stateful=False,name="Layer_9"))(g1)
i1=(tf.compat.v1.keras.layers.Conv1D(filters=2, kernel_size=1, strides=1, padding='valid', data_format="channels_last", dilation_rate=1, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(0.01),name="Layer_10"))(h1)
batch_31 = tf.compat.v1.keras.layers.BatchNormalization()(i1)
y2=tf.compat.v1.keras.layers.Activation('tanh')(batch_31)
j1=(tf.compat.v1.keras.layers.Flatten(data_format="channels_last",name="Layer_11"))(y2)
k1=(tf.compat.v1.keras.layers.Dense(units=1,activation='tanh',kernel_initializer='glorot_uniform',kernel_regularizer=tf.keras.regularizers.l2(0.01),name="Layer_12"))(j1)
l1=(tf.compat.v1.keras.layers.Dense(units=1,activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=tf.keras.regularizers.l2(0.01),name="O_Layer"))(k1)

total = tf.keras.models.Model(inputs=[Model_input],outputs=[l0,l1])#g
total.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MAE,metrics=[tf.keras.metrics.MAE])
print(total.summary())  
epochs=1000
#p=tf.compat.v1.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.0001,verbose=1,mode='min',restore_best_weights=True,patience=400)
q=tf.compat.v1.keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.1,patience=1000,verbose=1,mode='min',min_delta=0.0001,cooldown=0,min_lr=0)
r=tf.compat.v1.keras.callbacks.TensorBoard()
total.fit(dataset,epochs=epochs,verbose=2,shuffle=False,steps_per_epoch=45,callbacks=[q,r],validation_data=val_dataset,validation_freq=1,validation_steps=18)
prediction1,prediction2 = total.predict(submission)#prediction

#prediction = prediction.reshape(-1)
#sample_submission[:,1] = prediction
#submission = pandas.DataFrame(sample_submission,columns=['seg_id','time_to_failure'])
#submission.to_csv("Submission2.csv",index=None,sep=",")

sample_submission[:,1] = prediction1.reshape(-1)
submission = pandas.DataFrame(sample_submission,columns=['seg_id','time_to_failure'])
submission.to_csv("Submission3.csv",index=None,sep=",")

sample_submission[:,1] = prediction2.reshape(-1)
submission = pandas.DataFrame(sample_submission,columns=['seg_id','time_to_failure'])
submission.to_csv("Submission4.csv",index=None,sep=",")

exit(0)