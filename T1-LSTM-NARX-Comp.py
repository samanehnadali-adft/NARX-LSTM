#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      ADFT014-RD
#
# Created:     16/08/2021
# Copyright:   (c) ADFT014-RD 2021
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization, Flatten, Dropout, Bidirectional
from fireTS.models import NARX
from datetime import datetime

df = pd.read_csv(r'C:\Users\ADFT014-RD\Documents\PREDICTION\TEMPERATURE-DATA -2.csv')

print(df.head())
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['DateTime_diff'] = df['DateTime'].diff()
df['minutes'] = df['DateTime_diff'].apply(lambda x: x.seconds/60.)
print(df['minutes'].value_counts())
df[df['minutes']>=10]["minutes"].plot.hist(bins = 100)
plt.show()

df.plot.scatter("minutes", "Setpoint")

plt.show()

print(df['RH'].describe())
df.plot('DateTime', 'RH')
df.plot('DateTime', 'RoomTemp')
df.plot('DateTime', 'Setpoint')
plt.show()


df['year'] = df['DateTime'].apply(lambda x: x.year)
df['month'] = df['DateTime'].apply(lambda x: x.month)
df['day'] = df['DateTime'].apply(lambda x: x.day)
df['hour'] = df['DateTime'].apply(lambda x: x.hour)
df['minute'] = df['DateTime'].apply(lambda x: x.minute)


cols_to_consider = ['year','month', 'day', 'hour', 'minute', 'RH', 'Fan', 'Swing','RoomTemp', 'Setpoint']
df_train = df[cols_to_consider]
print(df_train.head())

X_cols = ['year', 'month', 'day', 'hour', 'minute', 'RH', 'Fan', 'Swing', 'RoomTemp']
y_col = 'Setpoint'

X = np.array(df_train[X_cols])
y = np.vstack(df_train[y_col])
vals = np.concatenate((y, X), axis = 1)
##vals_reframed = series_to_supervised(vals)
##cols_filtered = [c for c in vals_reframed.columns if c.endswith('(t-1)')]
##vals_reframed_arr = vals_reframed[cols_filtered].values
##X = vals_reframed_arr[:,1:]
##y = vals_reframed_arr[:,:1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, shuffle = False)


def plot_pred(pred, act, x):
    datelis = []
    for i in range(x.shape[0]):
        tmp_date = datetime(int(x[i,0]), int(x[i,1]),int(x[i,2]),int(x[i,3]), int(x[i,4]))
        datelis.append(tmp_date)
    p1, = plt.plot(datelis, act)
    p2, = plt.plot(datelis, pred)
    plt.legend([p1,p2], ["Actual", "Predicted"])
    plt.ylabel("Celcius")
    plt.xlabel("Date")
    plt.show()


# ===============================================NARX===============================================================
# ---------------------------------------------Linear Regression----------------------------------------------------

mdl_l1_p_lr = NARX(LinearRegression(),
                auto_order=2, exog_order = [1]*X_train.shape[1])
mdl_l1_p_lr.fit(X_train, y_train.ravel())
y_pred = mdl_l1_p_lr.forecast(X_train, y_train, step=len(y_test), X_future = X_test[:-1, :])
narx_lr_mse = mean_squared_error(y_test, y_pred)
print(narx_lr_mse)

narx_lr_mae = mean_absolute_error(y_test, y_pred)
print(narx_lr_mae)
plot_pred(y_pred, y_test, X_test)


#--------------------------------Random Forest----------------------------------------------------------------------

mdl_l1_p_rf = NARX(RandomForestRegressor(n_estimators = 100, random_state=0),
                auto_order=2, exog_order = [1]*X_train.shape[1])
mdl_l1_p_rf.fit(X_train, y_train)

y_pred = mdl_l1_p_rf.forecast(X_train, y_train, step=len(y_test), X_future = X_test[:-1, :])
narx_rf_mse = mean_squared_error(y_test, y_pred)
print(narx_rf_mse)
narx_rf_mse = mean_squared_error(y_test, y_pred)
print(narx_rf_mse)
narx_rf_mae = mean_absolute_error(y_test, y_pred)
print(narx_rf_mse)
plot_pred(y_pred, y_test, X_test)

#-------------------------------Gradient Boosting--------------------------------------------------
mdl_l1_p_gb = NARX(GradientBoostingRegressor(n_estimators = 100),
                auto_order=2, exog_order = [1]*X_train.shape[1])
mdl_l1_p_gb.fit(X_train, y_train)
y_pred = mdl_l1_p_gb.forecast(X_train, y_train, step=len(y_test), X_future = X_test[:-1, :])
narx_gb_mse = mean_squared_error(y_test, y_pred)
print(narx_gb_mse)
narx_gb_mae = mean_absolute_error(y_test, y_pred)
print(narx_gb_mae)
plot_pred(y_pred, y_test, X_test)

#================================LSTM network============================================================

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
model = Sequential()
model.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(loss='mean_absolute_error', optimizer='adam')
hist = model.fit(X_train, y_train, epochs = 50, validation_data = (X_test, y_test), verbose = 1, shuffle = False)
plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='test')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

y_pred = model.predict(X_test)
lstm_mse = mean_squared_error(y_test, y_pred)
print(lstm_mse)

lstm_mae = mean_absolute_error(y_test, y_pred)
print(lstm_mae)

print(y_test.shape)
print(X_test.shape)

plot_pred(y_pred, y_test[:,0], X_test[:,0,:])


#-------------------------------Saving the errors of the models to an excel file--------------------------------------

mod_list = ["NARX - Linear Regression",
           "NARX - Random Forest Regression",
           "NARX - Gradient Boosting Regression",
           "LSTM"]
mse_list = [narx_lr_mse, narx_rf_mse, narx_gb_mse,
           lstm_mse]
mae_list = [narx_lr_mae, narx_rf_mae, narx_gb_mae,
           lstm_mae]

err_df = pd.DataFrame()
err_df['Model'] = mod_list
err_df['Mean Squared Error'] = mse_list
err_df['Mean Absolute Error'] = mae_list
err_df=err_df.sort_values(by = 'Mean Absolute Error', ascending = True).reset_index(drop = True)
writer = pd.ExcelWriter('C:\\Users\\ADFT014-RD\\Documents\\PREDICTION\\Errors.xlsx', engine = 'xlsxwriter', options={'constant_memory': True})
err_df.to_excel(writer, sheet_name='Sheet1', index = False)
worksheet = writer.sheets['Sheet1']
writer.save()
writer.close()