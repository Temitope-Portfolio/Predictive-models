# Bitcoin Price Predictive Model
Date : 5/2/2022

## Why Price Prediction on Bitcoin?
Carrying out a price predictive model on Bitcoin with its volatility's nature, it is imperative to be able to predict its next move with the data available and check the accuracy of the model.

## Explaining Bitcoin Present Price Prediction
The relevant libraries were imported and loading the real data from yahoo finance API with the starting date from 2018,1,1 and ending date to be now -the current day- which is the time frame for the training data. 
Later went to prepare the data for the Neural network and scaled down the data within 0 and 1 so that Neural network will work better with it.


Next we went on to choice a number for the prediction days. The prediction days are going to be the number of days we're going to base our prediction on. We look at the number of past x_days e.g 60 days and then we predict one day in the future, hence, we look at 60 days then we predict the 61st day.
Then we neeed to prepare the training data. We need to have a x_data and y_data where y_data is going to be the result of the prediction. This is Supervised learning, we're saying we want the x_data to be the 60 days (actual value) data which is more concised and the y_data to be the result of the prediction.


Next, we're going to build a Neural network which is going to be the model that we're going to use for the prediction. We're going to equate model to Sequential() and add LSTM layers (Long Short-Term Memory) and dropout layers. LSTM layers are going to be recurrent layers which we are going to use to memorise data since we're dealing with sequencial data which has day 1, day 2, day 3 etc. Those layers are powerful because they are specialised on that sort of data and LSTM are memorising the crucial information by feeding the data back into the Neural network, while the dropout layer is to prevent overfitting. Then we're going to compile them all and train the model with 25 epochs.


Next we're going to test the model. For this, we need to specify a time frame for the testing data, load some testing data, and then compare the test results, the prediction result on the test data with the actual result. Now we're going to load data from yahoo finance api but with a new date for the test_start but same date for the end (end of the prediction which is the current date 'now'). 


After, we're going to plot both the actual price and the predicted price in the matplotlib with different colors so we can differenciate the prices and give it a title.  


![Screenshot 06-02-2022 122003](https://user-images.githubusercontent.com/81313873/152678397-a1d7a201-915e-49eb-822b-eda37c63a504.jpg)


The prediction curve looks good, even if they look almost alike, we need to keep in mind that we're only predicting one day based on 60 days. So we need to feed it 60 days data in order for us to be able to predict the 61st day. If we're to predict the 62nd day, we would need to feed it 59 days data and train the model.

## Explaining Bitcoin 30 days Price Prediction

If we want to predict actual days in the future not just looking at the past performance, we need to look at the actual prediction for that particular day. In this case, we are looking at 30 days prediction
