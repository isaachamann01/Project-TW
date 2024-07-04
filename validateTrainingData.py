import pandas as pd
import numpy as np
import tensorflow as tf

# RNN uses previous 50 close prices to predict the average over the next 5 close prices.
previous_timesteps = 50

def import_csv(filename):
    df = pd.read_csv(filename)
    selected_columns = ['Open time', 'Close']
    df = df[selected_columns]
    return df

def generate_rolling_averages(df):
    df['5_unit_avg'] = df['Close'].rolling(window=5).mean()
    df['20_unit_avg'] = df['Close'].rolling(window=20).mean()
    df['60_unit_avg'] = df['Close'].rolling(window=60).mean()
    df['10080_unit_avg'] = df['Close'].rolling(window=10080).mean()
    return df

def generate_numpy_array(df, timefrom, timeto):

    # We take the df provided and give a numpy array with shape (dflength, 50, 6)
    # Df Length is the length of the dataframe, and the 50 is the 50 entries before
    # 1st column in the df is the time and the next 5 are the close + rolling averages.

    # Find index of time from and time to
    timefrom = pd.to_datetime(timefrom)
    timeto = pd.to_datetime(timeto)
    # convert to unix time milliseconds
    timefrom = timefrom.timestamp() * 1000
    timeto = timeto.timestamp() * 1000

    timefrom_index = df[df['Open time'] == timefrom].index[0]
    timeto_index = df[df['Open time'] == timeto].index[0]

    print (timefrom_index)
    print (timeto_index)

    # Assuming df is your DataFrame and it has 6 columns
    dfLength = len(df[timefrom_index:timeto_index])

    # Initialize the 3D array
    array_3d = np.zeros((dfLength, 50, 6))

    # Fill the 3D array
    for i in range(timefrom_index, timeto_index):
        # Determine the start index for slicing the DataFrame, ensuring it's not negative
        start_idx = max(i - 49, 0)
        # Slice the DataFrame to get up to 50 rows ending with the current row
        temp_df = df.iloc[start_idx:i+1]
        #print(temp_df)
        # If there are fewer than 50 rows, pad the beginning with zeros
        if len(temp_df) < 50:
            padding = np.zeros((50 - len(temp_df), 6))
            temp_array = np.vstack((padding, temp_df.to_numpy()))
        else:
            temp_array = temp_df.to_numpy()
            # Assign this 2D array to the corresponding "page" in the 3D array
            array_3d[i-timefrom_index, :, :] = temp_array
    return array_3d

def remove_time(numpy_array):
    return numpy_array[:, :, 1:]

def normalise_numpy_array(array_3d):
    for i in range(5):
        min_val = np.min(array_3d[:, :, i])
        max_val = np.max(array_3d[:, :, i])
        array_3d[:, :, i] = (array_3d[:, :, i] - min_val) / (max_val - min_val)
    return array_3d

def fit_to_model(array_2d, model):
    # 2D array is the array of the last 50 close prices
    # We need to fit this to the model and get the prediction

    array_2d = array_2d.reshape(-1, previous_timesteps, 5)
    # print(array_2d)
    # print(array_2d.shape)
    prediction = model.predict(array_2d)
    return prediction

def check_prediction(current, prediction, position):
    # Get percentage difference between current and prediction
    percentage_difference = (prediction - current) / current

    # If percentage difference is greater than 0.002, return buy
    # Else return close
    if percentage_difference > 0.002:
        if position == 'buy':
            return 'hold'
        else:
            return 'buy'
    else:
        return 'close'
    
def check_ROI(array_3d):
    currententry = 0
    currentROI = 0
    position = 'close'
    for i in range(array_3d.shape[0]):
        prediction = check_prediction(array_3d[i, 0, 0], array_3d[i, 0, 5], position)
        if prediction == 'buy':
            position = 'buy'
            currententry = array_3d[i, 0, 0]
        elif prediction == 'close':
            if position == 'buy':
                position = 'close'
                currentROI += (array_3d[i, 0, 0] - currententry) / currententry + currentROI * (array_3d[i, 0, 0] - currententry) / currententry
    return currentROI
        
    
# Import BTCUSDT.h5 current folder.
# print tensorflow version
print(tf.__version__)
#model = tf.keras.models.load_model('BTCUSDT.h5')
import tensorflow as tf
from tensorflow.keras import datasets, layers, models,losses

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

x_length = 5
seq_length = 50

model = models.Sequential()
model.add(tf.keras.Input(shape=(seq_length, x_length)))  # Specify the input shape here
model.add(layers.LSTM(100, return_sequences=True))
model.add(layers.LSTM(100, return_sequences=False))
model.add(layers.Dense(50))
model.add(layers.Dense(1))  # Output layer with future_steps units

model.compile(optimizer='adam',
              loss=losses.Huber(),
              metrics=[losses.MeanAbsoluteError()])

# now load the weights from BTCUSD.keras
model.load_weights('BTCUSDT.keras')


#Import the CSV file
df = import_csv('BTCUSDT.csv')

# Generate rolling averages
df = generate_rolling_averages(df)

# print(df[df['Open time'] == 1706227200000])
#print (df)

# Generate numpy array
array_3d = generate_numpy_array(df, '2021-01-01 00:00:00', '2021-01-02 00:00:00')

print(array_3d.shape)

#Remove time from numpy array
array_3d = remove_time(array_3d)

print(array_3d.shape)

# # Normalise the numpy array
array_3d = normalise_numpy_array(array_3d)

print(array_3d.shape)

# # starting_cost = 100
prediction_array = np.pad(array_3d, ((0, 0), (0, 0), (0, 1)), mode='constant')
# #Fit the model for our chosen sample.
for i in range(array_3d.shape[0]):
    # the prediction
    prediction = fit_to_model(array_3d[i,:,:], model)
    prediction_array[i, :, 5] = prediction[0]

print(prediction_array    )

# # Check prediction
# # ROI = check_ROI(array_3d)
# # print(ROI)

# # Fit the model for our chosen sample.
# for i in range(array_3d.shape[0]):
#     # the prediction
#     array_3d[:,:,6] = check_prediction(array_3d[i,:,:], model)

# # Now we want to graph the stock price during this time and add a green dot when we buy and a red dot when we close.

# import matplotlib.pyplot as plt

# # Get the stock prices
# stock_prices = array_3d[:, 0, 0]

# # Create the plot
# plt.plot(stock_prices)

# # Add green dots for buy positions
# buy_positions = np.where(array_3d[:, 0, 6] == 'buy')[0]
# plt.scatter(buy_positions, stock_prices[buy_positions], color='green', marker='o')

# # Add red dots for sell positions
# sell_positions = np.where(array_3d[:, 0, 6] == 'close')[0]
# plt.scatter(sell_positions, stock_prices[sell_positions], color='red', marker='o')

# # Show the plot
# plt.show()


        



