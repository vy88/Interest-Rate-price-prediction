import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
vvv = pd.read_csv(r"D:\data science\excel files ds\credit_risk_dataset.csv")
vvv
a = vvv["loan_status"]==0
df = vvv.loc[~a]
df
df.isnull().sum()
df = df.dropna()
del df["loan_status"]
df

import seaborn as sns
sns.boxplot(x = "person_income",data = df)   # outlier find by boxplot,boxplot is best
plt.show()
Q1 = df["person_income"].quantile(0.25)   # 25 % data , can see value in describe function
Q3 = df["person_income"].quantile(0.75)   # 75 % data
Q1
Q3
IQR = Q3-Q1
min_range = Q1 - (1.5*IQR)
max_range = Q3 + (1.5*IQR)
min_range,max_range
df = df[df["person_income"]<=max_range]
df.shape
sns.distplot(df["person_income"])     # outlier find using distribution plot
plt.show()
sns.boxplot(x = "loan_int_rate",data = df)   # outlier find by boxplot,boxplot is best
plt.show()


Q1 = df["loan_int_rate"].quantile(0.25)   # 25 % data , can see value in describe function
Q3 = df["loan_int_rate"].quantile(0.75)   # 75 % data
Q1
Q3
IQR = Q3-Q1
min_range = Q1 - (1.5*IQR)
max_range = Q3 + (1.5*IQR)
min_range,max_range
df = df[df["loan_int_rate"]<=max_range]
df.shape
sns.distplot(df["loan_int_rate"])     # outlier find using distribution plot
plt.show()

sns.boxplot(x = "cb_person_cred_hist_length",data = df)   # outlier find by boxplot,boxplot is best
plt.show()
Q1 = df["cb_person_cred_hist_length"].quantile(0.25)   # 25 % data , can see value in describe function
Q3 = df["cb_person_cred_hist_length"].quantile(0.75)   # 75 % data
Q1
Q3
IQR = Q3-Q1
min_range = Q1 - (1.5*IQR)
max_range = Q3 + (1.5*IQR)
min_range,max_range
df = df[df["cb_person_cred_hist_length"]<=max_range]
df.shape
sns.distplot(df["cb_person_cred_hist_length"])     # outlier find using distribution plot
plt.show()

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

df["cb_person_default_on_file"]  = encoder.fit_transform(df["cb_person_default_on_file"])

df


X = df.drop(columns = "loan_int_rate")  # dividing the data into dependent and independent
y = df["loan_int_rate"]
X


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer([('encode',OneHotEncoder(),[2,4,5])],remainder = "passthrough")
X = ct.fit_transform(X)
X

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.15,random_state = 0)
print(X_test)

import tensorflow as tf

print(X_train.shape)

print(y_train.shape)

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2
ann = Sequential()
ann.add(Input(shape=(24,)))

ann.add(Dense(units = 24,activation = "relu",kernel_regularizer = regularizers.L2(0.5)))
ann.add(Dense(units = 24,activation = "relu",kernel_regularizer = regularizers.L2(0.5)))


ann.add(Dense(units =1,activation = "relu"))

ann.compile(optimizer= SGD(learning_rate = 0.000001), loss="mse", metrics=["mse"])
model = ModelCheckpoint("best_ann.keras",save_best_only = True)
history = ann.fit(X_train,y_train,batch_size =10,epochs=5000,validation_data=(X_test,y_test),callbacks = [model])

from matplotlib import pyplot as plt
plt.plot(history.history["val_loss"])
plt.plot(history.history['loss'])
plt.legend(["Train loss","Val loss"])
plt.show()

y_pred = ann.predict(X_test)
print(y_pred)

from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))
