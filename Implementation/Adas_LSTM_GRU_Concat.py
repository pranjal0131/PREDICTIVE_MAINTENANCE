#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install klib')


# In[3]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import klib
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


data = pd.read_csv('PredictiveManteinanceEngineTraining.csv')
data = data.drop('id', axis=1)
data.head()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


plt.figure(figsize=(20, 18))
klib.corr_plot(data)


# In[8]:


klib.dist_plot(data)


# In[9]:


klib.corr_mat(data)


# In[10]:


klib.data_cleaning(data)


# In[11]:


df = klib.convert_datatypes(data)


# In[12]:


df.info()


# In[13]:


klib.corr_plot(data, target='RUL')


# In[14]:


klib.corr_plot(data, target='label1')


# In[15]:


klib.corr_plot(data, target='label2')


# In[12]:


klib.corr_plot(data, target='cycle')


# In[13]:


data = data.astype({"label1":'category', "label2":'category'})
data.info()


# In[14]:


sns.set_style("whitegrid")
sns.countplot(x='label1', data=data)


# In[15]:


sns.set_style("whitegrid")
sns.countplot(x='label2', data=data)


# In[16]:


data['label2'].value_counts()


# In[17]:


data['label1'].value_counts()


# In[18]:


data.plot(kind="scatter", x="s12", y="s7")


# In[19]:


sns.FacetGrid(data, hue="label2", palette="husl", size=5)    .map(plt.scatter, "s12", "s7")    .add_legend()


# In[20]:


sns.FacetGrid(data, hue="label2", palette="husl", size=5)    .map(plt.scatter, "s21", "s20")    .add_legend()


# In[21]:


sns.FacetGrid(data, hue="label2", palette="husl", size=5)    .map(plt.scatter, "s12", "s20")    .add_legend()


# In[22]:


sns.FacetGrid(data, hue="label2", palette="husl", size=5)    .map(plt.scatter, "s21", "s7")    .add_legend()


# In[23]:


sns.boxplot(x="label1", y="s21", palette="husl", data=data)


# In[24]:


sns.boxplot(x="label1", y="s12", palette="husl", data=data)


# In[25]:


sns.boxplot(x="label2", y="s7", palette="husl", data=data)


# In[26]:


sns.violinplot(x="label1", y="s12", palette="husl", data=data)


# In[27]:


sns.violinplot(x="label2", y="s12", palette="husl", data=data)


# In[28]:


# sns.pairplot(data, hue="label2", palette="husl", size=3)


# In[71]:


import keras
from keras.layers.core import Activation
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, LSTM, GRU,Input,Concatenate
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint


# In[30]:


data


# In[31]:


import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Setting seed for reproducability
np.random.seed(1234)  
PYTHONHASHSEED = 0
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
get_ipython().run_line_magic('matplotlib', 'inline')


# In[32]:


# read training data 
train_df = pd.read_csv('PM_train.txt', sep=" ", header=None)


# In[33]:


train_df


# In[34]:


train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                     's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                     's15', 's16', 's17', 's18', 's19', 's20', 's21']
train_df


# In[35]:


# read ground truth data
truth_df = pd.read_csv('PM_truth.txt', sep=" ", header=None)
truth_df


# In[36]:


truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)


# In[37]:


# truth_df.columns[[1]]


# In[38]:


# read test data
test_df = pd.read_csv('PM_test.txt', sep=" ", header=None)
test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                     's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                     's15', 's16', 's17', 's18', 's19', 's20', 's21']


# In[39]:


train_df = train_df.sort_values(['id', 'cycle'])
train_df.head()


# In[40]:


train_df.groupby('id')['cycle'].max()


# In[ ]:





# In[ ]:





# In[41]:


rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
train_df = train_df.merge(rul, on=['id'], how='left')
train_df['RUL'] = train_df['max'] - train_df['cycle']
train_df.drop('max', axis=1, inplace=True)
train_df.head()


# In[42]:


w0 = 15
w1 = 30
train_df['label1'] = np.where(train_df['RUL'] <= w1, 1, 0)
train_df['label2'] = train_df['label1']
train_df.loc[train_df['RUL'] <= w0, 'label2'] = 2
train_df.head()


# In[43]:


train_df['cycle_norm'] = train_df['cycle']
col_normalize = train_df.columns.difference(['id','cycle','RUL','label1','label2'])
col_normalize


# In[44]:


min_max_scaler = MinMaxScaler()
norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[col_normalize]),
                            columns=col_normalize,
                            index=train_df.index)
joint_df = train_df[train_df.columns.difference(col_normalize)].join(norm_train_df)
train_df = joint_df.reindex(columns = train_df.columns)
train_df.head()


# In[45]:


test_df['cycle_norm'] = test_df['cycle']
norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[col_normalize]),
                           columns=col_normalize,index=test_df.index)
test_join_df = test_df[test_df.columns.difference(col_normalize)].join(norm_test_df)
test_df = test_join_df.reindex(columns=test_df.columns)
test_df = test_df.reset_index(drop=True)
test_df.head()


# In[46]:


test_df.head(5)


# In[47]:


rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
rul


# In[48]:


rul.columns = ['id','max']
rul


# In[49]:


truth_df


# In[50]:


truth_df.columns = ['more']
truth_df

truth_df['id'] = truth_df.index + 1
truth_df

truth_df['max'] = rul['max'] + truth_df['more']
truth_df

truth_df.drop('more',axis=1, inplace=True)


# In[51]:


truth_df


# In[52]:


# generate RUL for test data
test_df = test_df.merge(truth_df, on=['id'], how='left')
test_df['RUL'] = test_df['max'] - test_df['cycle']
test_df.drop('max', axis=1, inplace=True)
test_df.head()


# In[53]:


# generate label columns w0 and w1 for test data
test_df['label1'] = np.where(test_df['RUL'] <= w1, 1, 0 )
test_df['label2'] = test_df['label1']
test_df.loc[test_df['RUL'] <= w0, 'label2'] = 2
test_df.head()


# In[54]:


sequence_length = 50
# preparing data for visualizations 
# window of 50 cycles prior to a failure point for engine id 3
engine_id3 = test_df[test_df['id'] == 3]
engine_id3_50cycleWindow = engine_id3[engine_id3['RUL'] <= engine_id3['RUL'].min() + 50]
cols1 = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']
engine_id3_50cycleWindow1 = engine_id3_50cycleWindow[cols1]
cols2 = ['s11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
engine_id3_50cycleWindow2 = engine_id3_50cycleWindow[cols2]


# In[55]:


# plotting sensor data for engine ID 3 prior to a failure point - sensors 1-10 
ax1 = engine_id3_50cycleWindow1.plot(subplots=True, sharex=True, figsize=(20,20))


# In[56]:


# plotting sensor data for engine ID 3 prior to a failure point - sensors 11-21 
ax2 = engine_id3_50cycleWindow2.plot(subplots=True, sharex=True, figsize=(20,20))


# In[57]:


def gen_sequence(id_df,seq_length,seq_cols):
#     print(id_df)
    data_array = id_df[seq_cols].values
#     print(data_array)
#     input()
    num_elements = data_array.shape[0]
#     print(data_array.shape[0])
    for start,stop in zip(range(0,num_elements-seq_length), range(seq_length,num_elements)):
        yield data_array[start:stop,:]


# In[58]:


# pick the feature columns 
sensor_cols = ['s' + str(i) for i in range(1, 22)]
# sensor_cols = ['s2', 's3', 's4', 's6', 's8', 's9', 's10', 's11', 's13', 's14', 's15', 's17']


# In[59]:


sequence_cols = ['setting1', 'setting2','setting3','cycle_norm']
# sequence_cols = ['setting1', 'setting2','cycle_norm']
sequence_cols.extend(sensor_cols)
sequence_cols


# In[60]:


(train_df[train_df['id']==id])[sequence_cols[1]].values


# In[61]:


# generator for the sequences
seq_gen = (list(gen_sequence(train_df[train_df['id']==id], sequence_length, sequence_cols)) 
           for id in train_df['id'].unique())
# seq_gen


# In[62]:


# generate sequences and convert to numpy array
seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
seq_array.shape


# In[63]:


seq_array[0].shape


# In[64]:


# function to generate labels
def gen_labels(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length:num_elements, :]


# In[65]:


# generate labels
label_gen = [gen_labels(train_df[train_df['id']==id], sequence_length, ['label1']) 
             for id in train_df['id'].unique()]
label_array = np.concatenate(label_gen).astype(np.float32)
label_array.shape


# In[145]:


filepath="weights.best.hdf5"
# early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')


# In[67]:


# build the network
nb_features = seq_array.shape[2]
nb_out = label_array.shape[1]

model = Sequential()

model.add(GRU(
         input_shape=(sequence_length, nb_features),
         units=100,
         return_sequences=True))
model.add(Dropout(0.2))

model.add(GRU(
          units=100,
          return_sequences=True))
# model.add(Dropout(0.2))
model.add(Dropout(0.2))

model.add(GRU(
          units=50,
          return_sequences=True))
model.add(Dropout(0.2))

model.add(GRU(
          units=50,
          return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=nb_out, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())


# In[146]:


def best_model():
    input1 = Input(shape=(sequence_length,nb_features),name='input1',dtype='float32')
    input2 = Input(shape=(sequence_length,nb_features),name='input2',dtype='float32')
    
    lstm1 = LSTM(units=200, return_sequences=True)(input1)
    lstm1 = Dropout(0.2)(lstm1)
    lstm1 = LSTM(units=100, return_sequences=True)(lstm1)
    lstm1 = Dropout(0.2)(lstm1)
    lstm1 = LSTM(units=50, return_sequences=False)(lstm1)
    lstm1 = Dropout(0.2)(lstm1)
    
    lstm2 = GRU(units=200, return_sequences=True)(input2)
    lstm2 = Dropout(0.2)(lstm2)
    lstm2 = GRU(units=100, return_sequences=True)(lstm2)
    lstm2 = Dropout(0.2)(lstm2)
    lstm2 = GRU(units=50, return_sequences=False)(lstm2)
    lstm2 = Dropout(0.2)(lstm2)
    
    comb = Concatenate(axis=1)([lstm1,lstm2])
    pred = Dense(units=nb_out, activation='sigmoid', name='output')(comb)
    model = Model(inputs=[input1,input2], outputs=pred)
    
    return model


# In[147]:


model = best_model()
model.summary()


# In[148]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[149]:


get_ipython().run_cell_magic('time', '', "# fit the network\nmodel.fit([seq_array,seq_array], label_array, epochs=20, batch_size=200, validation_split=0.05, verbose=1,\n          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto'), checkpoint])")


# In[150]:


model.load_weights("weights.best.hdf5")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[151]:


# training metrics
scores = model.evaluate([seq_array,seq_array], label_array, verbose=1, batch_size=200)
print('Accurracy: {}'.format(scores[1]))


# In[152]:


# make predictions and compute confusion matrix
y_pred = model.predict([seq_array,seq_array],verbose=1, batch_size=200)
y_pred[:10]


# In[153]:


y_prediction = []
for item in y_pred:
    if item < 0.5:
        y_prediction.append(0)
    else:
        y_prediction.append(1)
        
print(y_pred.shape, len(y_prediction))


# In[154]:


y_true = label_array
print('Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')
cm = confusion_matrix(y_true, y_prediction)
cm


# In[155]:


# compute precision and recall
precision = precision_score(y_true, y_prediction)
recall = recall_score(y_true, y_prediction)
print( 'precision = ', precision, '\n', 'recall = ', recall)


# In[156]:


seq_array_test_last = [test_df[test_df['id']==id][sequence_cols].values[-sequence_length:] 
                       for id in test_df['id'].unique() if len(test_df[test_df['id']==id]) >= sequence_length]

seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)
seq_array_test_last.shape


# In[157]:


y_mask = [len(test_df[test_df['id']==id]) >= sequence_length for id in test_df['id'].unique()]


# In[158]:


label_array_test_last = test_df.groupby('id')['label1'].nth(-1)[y_mask].values
label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0],1).astype(np.float32)
label_array_test_last.shape


# In[159]:


print(seq_array_test_last.shape)
print(label_array_test_last.shape)


# In[160]:


# test metrics
scores_test = model.evaluate([seq_array_test_last,seq_array_test_last], label_array_test_last, verbose=2)
print('Accurracy: {}'.format(scores_test[1]))


# In[161]:


# make predictions and compute confusion matrix
y_pred_test = model.predict([seq_array_test_last,seq_array_test_last])


# In[162]:


y_prediction_test = []
for item in y_pred_test:
    if item < 0.5:
        y_prediction_test.append(0)
    else:
        y_prediction_test.append(1)
        
print(y_pred_test.shape, len(y_prediction_test))


# In[163]:



y_true_test = label_array_test_last
print('Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')
cm = confusion_matrix(y_true_test, y_prediction_test)
cm


# In[164]:


# compute precision and recall
precision_test = precision_score(y_true_test, y_prediction_test)
recall_test = recall_score(y_true_test, y_prediction_test)
f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)
print( 'Precision: ', precision_test, '\n', 'Recall: ', recall_test,'\n', 'F1-score:', f1_test )


# In[167]:


results_df = pd.DataFrame([[scores_test[1],precision_test,recall_test,f1_test],
                          [0.94, 0.952381, 0.8, 0.869565]],
                         columns = ['Accuracy', 'Precision', 'Recall', 'F1-score'],
                         index = ['LSTM-GRU',
                                 'Template Best Model'])
results_df


# In[166]:


loss_df = pd.DataFrame(model.history.history)
loss_df.plot(figsize=(10,8))


# In[ ]:




