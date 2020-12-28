#!/usr/bin/env python
# coding: utf-8

# ## SDG Attainability for African countries - V.3 - Final
# Objective :Use data science techniques to predict which countries can attain or not the SDG goals by 2030
# Help to take decision now (2020) on effort to be made by each countries
# 
# Data Source : 
#     >UN Open data () - Extract by continent (AFRICA) Format : XLS
#         https://unstats.un.org/sdgs/indicators/database
#         

# ## 1. INITIAL DATA EXPLORATION AND PROCESSING

# >### 1.1 DATA LOADING

# In[1]:


import pandas as pd

full_data = pd.read_excel ('D:\TO_BE_CLASSIFIED\Perso\Data Science\Expert Data Science - IBM\Certificate Capstone\DATA\SDGs All Goals - 2020.xlsx', 
                           sheet_name='data', dtype=None)


# In[2]:


full_data.dtypes


# In[4]:


full_data


# In[5]:


# Convert into Pandas Dataframe
full_data.shape
df = pd.DataFrame (full_data)

print (df.shape)
df.head(5)


# In[6]:


# Conversion of column Value to float 

from pandas.api.types import is_numeric_dtype
df.shape
df['Value'].astype('float')                                          ## Convert the column to float
print (df['Value'].apply(type))                                      ## Display the type of the Value column for each row
df['Value_Type'] = df['Value'].apply(lambda x: type(x).__name__)     
df['Value'] = df['Value'].astype('float')
df
is_numeric_dtype(df['Value'])
df['Value'].apply(type).value_counts()


# >### 1.2 DATA CONVERSION - PIVOT TABLE

# In[7]:


#
# Use the pivot_table function to convert the dataframe
# "Index" columns do NOT move
# Columns in the "columns" parameter become NEW columns
# The column "value" give the value
#

# Shape input = (280087 rows, 9 columns)
# Shape output = (15606 rows, 38 columns)

by_year_table_df = df.pivot_table(index = ["Goal", "Target", "Indicator", "SeriesCode", "SeriesDescription", 
                                                  "GeoAreaCode", "GeoAreaName"], 
                                                 columns = "TimePeriod", 
                                                 values  = "Value")


# In[8]:


by_year_table_df.shape


# In[9]:


by_year_table_df
# Note that the first 5 columns are in "group by" format 


# In[10]:


# To repeat for each row these "grouped by" columns, write the data in csv file 
          
by_year_table_df.to_csv (r'D:\TO_BE_CLASSIFIED\Perso\Data Science\Expert Data Science - IBM\Certificate Capstone\DATA\\export_dataframe.csv', header=True)


# In[11]:


# And read it again into a Pandas dataframe

final_data = pd.read_csv ('D:\TO_BE_CLASSIFIED\Perso\Data Science\Expert Data Science - IBM\Certificate Capstone\DATA\\export_dataframe.csv') 


# In[12]:


# Final data format 

print(final_data.shape)

final_data


# >### 1.3 DATA QUALITY : 
# - Drop non-relevent columns after their quality (so much NaN)
# - Choice of columns used as LABEL for the supervised algos : Columns 2018, 2019 and 2020. Note that at this point of time, columns 2019 and 2020 are not yet fed entirely.
# - Missing values
# 

# In[13]:


# Drop non used columns (1980 to 1999)
# Columns 2018, 2019 and 2020 will be used for testing (so, they are kept in the dataset)
#
# Shape : (17401 rows, 28 columns)

final_data = final_data.drop(columns=['1980', '1984','1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999'], axis=1)


# In[14]:


# Save the file in csv format
final_data.to_csv (r'D:\\TO_BE_CLASSIFIED\\Perso\\Data Science\\SDG Forcasting\\UN Docs\\export_final_data.csv', header=True)


# In[15]:


final_data.shape


# In[16]:


final_data


# In[17]:


#
# Count the NaN or null values in the dataframe (percentage)
#

r = final_data.shape[0]
c = final_data.shape[1] - 7 # Only datapooints columns : 2000 to 2020.
nan = final_data.isnull().sum().sum() 

print ('%s Rows X %s Columns = %s datapoints.' % (r, c, r * c) )
print ('Total number of NaN values : %s datapoints that means %.2f %%' % (nan, nan / (r * c) * 100 ))


# In[18]:


# 
# Null values repartition by SDGs objectives
#
#datapoints_by_goal = []
#nan_list = list()

nan_list = pd.DataFrame(columns=['Goal', 'Datapoints','Nan_percentage'],index=range(17))
             
for i in range (17):
    r = final_data[final_data['Goal']== i+1].shape[0]
    c = final_data.shape[1] - 7
    
    nan_by_goal = final_data[final_data['Goal']==i+1].isnull().sum().sum() 
    # datapoint_by_goal [i]  = r 
    
    # print ('%s Rows X %s Columns = %s datapoints.' % (r, c, r * c) )
    # print ('Total number of NaN values in Goal %d : %s datapoints that means %.2f %%' % (i + 1, nan_by_goal, nan_by_goal / (r * c) * 100 ))
    
    nan_list ['Goal'][i] = i + 1
    nan_list ['Datapoints'][i] = r * c
    nan_list ['Nan_percentage'][i] = round(nan_by_goal / (r * c) * 100,2)
    
nan_list


# In[20]:


#
# Plot and visualize the importance of the missing data compared with the available datapoints by goal
#

import matplotlib.pyplot as plt

nan_list.plot(x='Goal', y='Nan_percentage')

# nan_list.plot.barh(x='Goal', stacked=True);

# Interesting goals are the ones with less number of NaN : Goals 7,8,9,10 and 17
#

fig = plt.figure() # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

width = 0.4

nan_list.Datapoints.plot(kind='bar', color='red', ax=ax, width=width, position=1)
nan_list.Nan_percentage.plot(kind='bar', color='blue', ax=ax2, width=width, position=0)

ax.set_ylabel('Datapoints')
ax2.set_ylabel('NaN (%)')

plt.show()

fig.show()


# >### 1.4 FILL MISSING VALUES - LINEAR INTERPOLATION

# In[22]:


# Use a linear interpolation on data from 2000 to 2018 to fill MISSING VALUES
#
cols = ['2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017']
final_data[cols] = final_data[cols].interpolate(method="linear", limit_direction="both", axis=1)


# In[23]:


final_data


# In[24]:


#
# After linear interpolation
# Count again the NaN or null values in the dataframe (percentage)
#

r = final_data.shape[0]
c = final_data.shape[1]
nan = final_data.isnull().sum().sum() 

print ('%s Rows X %s Columns = %s datapoints.' % (r, c, r * c) )
print ('Total number of NaN values : %s datapoints that means %.2f %%' % (nan, nan / (r * c) * 100 ))


# In[25]:


# Check rows containing NaN value
# and drop them (because there is NO datapoint from 2000 to 2018)
# 
nan_df = final_data[final_data.isna().any(axis=1)]
print ('%s ROWS : No way to interpolate because no datapoints at all - to be dropped ' % nan_df.shape[0])


# In[26]:


# Drop these empty rows 
# Do NOT touch 2019 and 2020 columns even they have NaN values
#
final_data = final_data.dropna(subset=cols)


# In[27]:


# Finally, we get the cleaned dataset
final_data.shape


# In[28]:


final_data.reset_index(drop=True, inplace=True)


# In[29]:


final_data


# >### 1.5 DESCRIPTIVE STATISTICS AND VIZUALISATIONS

# In[30]:


df_211 = final_data.loc[final_data['Indicator'] == '2.1.1']


# In[31]:


print (df_211.shape)
df_211.head(50)


# In[32]:


import matplotlib.pyplot as plt

# Consider only an indicator (e.g. 1.1.1) to plot (Spagethi format)
#df = final_data.loc[final_data['Indicator'] == '2.1.1']

df = final_data.loc[(final_data["Indicator"] == '2.1.1') & (final_data["GeoAreaName"] == 'Burkina Faso')]
df.head(100)


# In[33]:


# Number of indicators to be consedered in the dataset
#
print('There are {} unique indicators.'.format(len(final_data['Indicator'].unique())))

# Number of countries in the dataset
#
print('There are {} unique countries.'.format(len(final_data['GeoAreaName'].unique())))


# In[34]:


from matplotlib import pyplot

df = df [cols]
ax = df.T.plot (kind='line')
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.
#ax.set_ylabel(cols)


# ## 2. CREATE THE MODEL - Model #1
# ### LSTM UNIVARIATE MULTI-STEPS (Vector Output Model)
#     As with one-step forecasting, a time series used for multi-step time series forecasting must be split into samples 
#     with input and output components.
#     Both the input and output components will be comprised of multiple time steps and may or may not have the same number 
#     of steps.
#     

# In[36]:


#
# Convert each line of the dataframe into list (time series) keeping just the columns 2000 to 2017 (Training set)
#

index_test = 1369
ts_list = final_data[cols].values.tolist()
ts_list [index_test]               ## This is the x-th line from the dataframe


# In[37]:


#
# multi-step data preparation: It aims to split each sequence (ts_list) into 2 lists X (input) and y (output)
# X (Input) has n_steps_in datapoints from ts_list.
# y (Output) has the next n_steps_out datapoints from ts_list
#

from numpy import array
 
# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)


# In[38]:


# Build test list from the columns 2018, 2019, 2020
test_list = final_data[['2018', '2019','2020']] 
print(test_list)

#
# Count the NaN or null values in the TEST LIST dataframe (percentage)
#

r = test_list.shape[0]
c = test_list.shape[1] 
nan = test_list.isnull().sum().sum() 

print ('BEFORE LINARIZATION : %s Rows X %s Columns = %s datapoints.' % (r, c, r * c) )
print ('BEFORE LINARIZATION : Total number of NaN values : %s datapoints that means %.2f %%' % (nan, nan / (r * c) * 100 ))

# Use a linear interpolation on data from 2000 to 2018 to fill MISSING VALUES
#
cols = ['2018','2019','2020'] 
test_list[cols] = test_list[cols].interpolate(method="linear", limit_direction="both", axis=1)

nan = test_list.isnull().sum().sum() 
print ('AFTER LINARIZATION : %s Rows X %s Columns = %s datapoints.' % (r, c, r * c) )
print ('AFTER LINARIZATION : Total number of NaN values : %s datapoints that means %.2f %%' % (nan, nan / (r * c) * 100 ))

test_list = test_list[cols].values.tolist()

test_list [index_test]


# In[39]:


# convert time series from "ts_list" to SEQUENCE

# choose a number of time steps INPUT and OUTPUT
n_steps_in, n_steps_out = 2, 3                     

# split into samples
X, y = split_sequence(ts_list [index_test], n_steps_in, n_steps_out)
# summarize the data
for i in range(len(X)):
    print(X[i], y[i])


# In[40]:


# The LSTM expects data to have a three-dimensional structure of [samples, timesteps, features] for the input data X, 
# and in this case, we want to have n_feature feature(s).
#
# Reshape from [samples, timesteps] into [samples, timesteps, features]
#
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
X[0], y[0]


# In[41]:


X, y


# In[42]:


# define model
from numpy import array
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_steps_out))

model.compile(optimizer='Adamax', loss='mse', metrics=['mse', 'accuracy'])

# model.compile(optimizer='RMSProp', loss='mse')


# In[43]:


from keras.utils.vis_utils import plot_model

#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#plot_model(model, show_shapes=True, show_layer_names=True)


# In[44]:


# fit model
#history = model.fit(X, y, epochs=50, verbose=0)

# Fit the model
history = model.fit(X, y,validation_split=0.33, epochs=50, verbose=0)       # batch_size=10, 
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# history = model.fit(X, y, epochs=50, batch_size=72, validation_data=(test_list[index_test], ????), verbose=2, shuffle=False)
# model.evaluate(np.asarray([np.zeros((2))]), np.asarray([np.zeros((12))]))


# In[46]:


# Make Prediction
#x_input = array([234.26035, 592.52705, 1235.32663, 991.14032])
x_input = array([17.6, 18.3])
x_input = x_input.reshape(1, n_steps_in, 1)
yhat = model.predict(x_input, verbose=0)
yhat = yhat.reshape(-1)
print(test_list[index_test], yhat)


# In[47]:


yhat


# In[48]:


#import necessary libraries
from sklearn.metrics import mean_squared_error
from math import sqrt

#calculate RMSE
rmse = sqrt(mean_squared_error(test_list[index_test], yhat)) 

print('Test RMSE: %.3f' % rmse)

