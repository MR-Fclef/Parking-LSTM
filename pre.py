import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# %matplotlib inline

# load the dataset
dataframe = read_csv('traindata-test.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataframetest = read_csv('test.csv', usecols=[1], engine='python', skipfooter=3)
datasetest = dataframetest.values
# transform int to float
dataset = dataset.astype('float32')
datasetest = datasetest.astype('float32')

plt.plot(dataset)
plt.show()

plt.plot(datasetest)
plt.show()

# X is the number of passengers at a given time (t) and Y is the number of passengers at the next time (t + 1).

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

# fix random seed for reproducibility
numpy.random.seed(7)


# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
datasetest = scaler.fit_transform(datasetest)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# use this function to  the train and test datasets for modeling
look_back = 10
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
newtestX, newtestY = create_dataset(datasetest, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.rprepareeshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
newtestX = numpy.reshape(newtestX,(newtestX.shape[0], newtestX.shape[1], 1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
newtestPredict = model.predict(newtestX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
newtestPredict = scaler.inverse_transform(newtestPredict)
newtestY = scaler.inverse_transform([newtestY])

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
newtestScore = math.sqrt(mean_squared_error(newtestY[0], newtestPredict[:,0]))
print('NewTest Score: %.2f RMSE' % (newtestScore))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2):len(dataset)+1, :] = testPredict

# shift newtest predictions for plotting
newtestPredictPlot = numpy.empty_like(datasetest)
newtestPredictPlot[:,:] = numpy.nan
newtestPredictPlot[look_back:len(newtestPredict)+look_back, :] = newtestPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

plt.plot(scaler.inverse_transform(datasetest))
plt.plot(newtestPredictPlot)
plt.show()

