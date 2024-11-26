# %%
"""
# Importing Neccessary Libraries
"""
# %%
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization,Dropout,Dense,Flatten,Conv1D
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import warnings
from matplotlib import gridspec
warnings.filterwarnings("ignore")


# %%
"""
# dataset information
"""

# %%
df = pd.read_csv('dataset/creditcard.csv', encoding= 'unicode_escape')

# %%
print('-------------------------------------------------------')
print('\nDataset First & Last 5 rows :\n',df.head())
print('\nDataset First & Last 5 rows :\n',df.tail())
print('\n-------------------------------------------------------')
print('\nIs Null : ',df.isnull().values.any())
print("\nNon-missing values: " + str(df.isnull().shape[0]))
print("\nMissing values: " + str(df.shape[0] - df.isnull().shape[0]))
print('\n-------------------------------------------------------')
print('\nDataset Shape (Rows & Columns) : ',df.shape)
print('\n-------------------------------------------------------')
print('\nClass Labels : ',df.Class.unique())
print('\n-------------------------------------------------------')
print('\nClass Label Counts :\n',df.Class.value_counts())
print('\n-------------------------------------------------------')
nf = df[df.Class==0]
f = df[df.Class==1]
print('\nClass 0 Samples : \n',nf)
print('Class 1 Samples : \n',f)
print('\n-------------------------------------------------------')
print('\nAbout Amount Attribute :\n',df["Amount"].describe())
print('\n-------------------------------------------------------')
non_fraud = len(df[df.Class == 0])
fraud = len(df[df.Class == 1])
fraud_percent = (fraud / (fraud + non_fraud)) * 100

print("\nTotal Number of Genuine transactions: ", non_fraud)
print("\nTotal Number of Fraud transactions: ", fraud)
print("\nTotal Percentage of Fraud transactions: {:.4f}".format(fraud_percent))
print('\n-------------------------------------------------------')


# %%
"""
# dataset Visualisation 1
"""

labels = ["Genuine", "Fraud"]
count_classes = df.value_counts(df['Class'], sort= True)
count_classes.plot(kind = "bar", rot = 0)
plt.title("Visualization of Labels")
plt.ylabel("Count")
plt.xticks(range(2), labels)
plt.show()


# %%
"""
# dataset Visualisation 2
"""

# plot the named features 
f, axes = plt.subplots(1, 2, figsize=(18,4), sharex = True)

amount_value = df['Amount'].values # values
time_value = df['Time'].values # values

sns.distplot(amount_value, hist=False, color="m", kde_kws={"shade": True}, ax=axes[0]).set_title('Distribution of Amount')
sns.distplot(time_value, hist=False, color="m", kde_kws={"shade": True}, ax=axes[1]).set_title('Distribution of Time')
plt.show()



"""
# dataset Visualisation 3
"""

# Correlation matrix

plt.figure(figsize=(20,8))
sns.heatmap(df.corr(), annot = False, cmap = 'copper')
plt.title('Correlation Heatmap of the Data', fontsize = 15)
plt.show()

# Plot histograms of each parameter 
df.hist(figsize = (20, 20))
plt.show()


# %%
"""
# Uneven class distribution
"""

# %%
print(df.Class.value_counts())

# %%
nf = df[df.Class==0]
f = df[df.Class==1]

# %%
"""
# Extracting random entries of class-0
# Total entries are 1.5* NO. of class-1 entries
"""

# %%
nf = nf.sample(1000)

# %%
"""
# Creating new dataframe
"""

# %%
data = nf


# %%
print(data.shape)




# %%
X = data.drop(['Class'],axis=1)
y=data['Class']



"""
# Train-Test Split
"""

# %%
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y)

# %%
print(X_train.shape,X_test.shape)

# %%
"""
# Applying StandardScaler to obtain all the features in similar range
"""

# %%
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# %%
y_train=y_train.to_numpy()
y_test=y_test.to_numpy()

# %%
"""
# Reshaping the input to 3D.
"""

# %%
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)

# %%
"""
# CNN model
"""
print('CNN Model....')
# %%
model=Sequential()
model.add(Conv1D(32,2,activation='relu',input_shape=X_train[0].shape))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv1D(64,2,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))

# %%
model.summary()

# %%
"""
# Compiling and Fiting
"""

# %%
model.compile(optimizer=Adam(learning_rate=0.0001),loss='binary_crossentropy',metrics=['accuracy'])

# %%
history = model.fit(X_train,y_train,epochs=20,validation_data=(X_test,y_test))

# %%
def plotLearningCurve(history,epochs):
  epochRange = range(1,epochs+1)
  plt.plot(epochRange,history.history['accuracy'])
  plt.plot(epochRange,history.history['val_accuracy'])
  plt.title('Model Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend(['Train','Validation'],loc='upper left')
  plt.show()

  plt.plot(epochRange,history.history['loss'])
  plt.plot(epochRange,history.history['val_loss'])
  plt.title('Model Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend(['Train','Validation'],loc='upper left')
  plt.show()

# %%
plotLearningCurve(history,20)

# %%
