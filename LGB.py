import pandas
import numpy
import lightgbm as lgbm 
import sys
import librosa
import scipy
import pywt
import sklearn
from tqdm import tqdm
from scipy import signal
from scipy import stats
from scipy import fftpack
from statsmodels.robust import mad
from sklearn.model_selection import KFold 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 
scaler = MinMaxScaler()
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore") 	
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
training = numpy.float32(data[:,0])
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
gaussian_noise = numpy.random.normal(0,0.5,window,seed)
data = numpy.zeros([q,n_feat])
for i in tqdm(range(0,q)):
    a = training[i,:]
    a = a + gaussian_noise
    a = a - numpy.median(a)
    coeff = pywt.wavedec(a,wavelet='db4',mode='per',level=1)
    sigma = mad(coeff[-1])
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
#data = scaler.fit_transform(data)
print("Training data has been prepared , moving on to preparing the testing data")
sample_submission = pandas.read_csv("sample_submission.csv",header='infer')
PATH = "test/"
submission_len = len(sample_submission)
sample_submission = sample_submission.to_numpy()
submission = numpy.empty([submission_len,n_feat])
del training
for i in tqdm(range(0,submission_len)):
    temp = pandas.read_csv(PATH+sample_submission[i,0]+".csv",header='infer',dtype=numpy.int16)
    temp = temp + gaussian_noise.reshape(-1,1)
    temp = temp - temp.median()
    temp = temp.to_numpy(dtype=numpy.float32)
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
#submission = scaler.fit_transform(submission)
prediction = numpy.zeros([submission_len,1])
folded = KFold(n_splits=n_fold,shuffle=True,random_state=seed)
folded = list(folded.split(numpy.arange(q)))
for fold_n,(train_index,val_index) in enumerate(folded):
    print('Fold',fold_n)
    training_data = lgbm.Dataset(data[train_index],label=output[train_index])
    val_data = lgbm.Dataset(data[val_index],label=output[val_index])
    params = {
    "objective":"fair",
    "num_leaves":10,
    "min_data_in_leaf":5,
    "boosting" :"gbdt",
    "learning_rate":0.005,
    "device_type":"gpu",
    "max_depth":-1,
    "bagging_fraction":0.5,
    "bagging_freq":1,
    "feature_fraction":0.8,
    "bagging_seed":0,
    "boost_from_average":True,
    "metric":"mae",
    "verbosity":-1,
    }
    model =lgbm.train(params=params,train_set=training_data,num_boost_round=10**5,valid_sets=[training_data,val_data],early_stopping_rounds=200,verbose_eval=10**4)
    a = model.predict(submission,num_iterations = model.best_iteration)
    prediction = prediction + a.reshape(-1,1)

print("Feature Importance")
axes = lgbm.plot_importance(model)
plt.show()


print("Another boosting tree which has to be rendered ")
graph = lgbm.create_tree_digraph(model)
graph.render(view=True)

model.save_model('LGBM.txt')
print("Done with saving and printing everything")

prediction = prediction/n_fold
prediction = prediction.reshape(-1)
sample_submission[:,1] = prediction
submission = pandas.DataFrame(sample_submission,columns=['seg_id','time_to_failure'])
submission.to_csv("Submission.csv",index=None,sep=",")