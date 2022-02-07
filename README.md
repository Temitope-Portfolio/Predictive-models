# Bitcoin Price Predictive Model
Date : 5/2/2022

## Why Price Prediction on Bitcoin?
Carrying out a price predictive model on Bitcoin with its volatility's nature, it is imperative to be able to predict its next move with the data available and check the accuracy of the model.

## Explaining Bitcoin Present Price Prediction
The relevant libraries were imported and loading the real data from yahoo finance API with the starting date from 2018,1,1 and ending date to be now -the current day- which is the time frame for the training data. 


![Screenshot 07-02-2022 231842](https://user-images.githubusercontent.com/81313873/152881811-850291aa-e4b7-42a1-8eac-e4d24dcea272.jpg)


Later went to prepare the data for the Neural network and scaled down the data within 0 and 1 so that Neural network will work better with it.


Next we're going to choose a number for the prediction days. The prediction days are going to be the number of days we're going to base our prediction on. We look at the number of past x_days e.g 60 days and then we predict one day in the future, hence, we look at 60 days then we predict the 61st day.
Then we neeed to prepare the training data. We need to have a x_data and y_data where y_data is going to be the result of the prediction. This is Supervised learning, we're saying we want the x_data to be the 60 days (actual value) data which is more concised and the y_data to be the result of the prediction.

![Screenshot 07-02-2022 232316](https://user-images.githubusercontent.com/81313873/152882297-76812fbe-2874-4aaf-9cad-cbf31942480d.jpg)



Next, we're going to build a Neural network which is going to be the model that we're going to use for the prediction. We're going to equate model to Sequential() and add LSTM layers (Long Short-Term Memory) and dropout layers. LSTM layers are going to be recurrent layers which we are going to use to memorise data since we're dealing with sequencial data which has day 1, day 2, day 3 etc. Those layers are powerful because they are specialised on that sort of data and LSTM are memorising the crucial information by feeding the data back into the Neural network, while the dropout layer is to prevent overfitting. Then we're going to compile them all and train the model with 25 epochs.

![Screenshot 07-02-2022 232436](https://user-images.githubusercontent.com/81313873/152882464-e0aa2367-dab5-4961-884b-e90edf7ed29c.jpg)

Next we're going to test the model. For this, we need to specify a time frame for the testing data, load some testing data, and then compare the test results, the prediction result on the test data with the actual result. Now we're going to load data from yahoo finance api but with a new date for the test_start but same date for the end (end of the prediction which is the current date 'now'). After, we're going to plot both the actual price and the predicted price in the matplotlib with different colors so we can differenciate the prices and give it a title.  

![Screenshot 07-02-2022 232606](https://user-images.githubusercontent.com/81313873/152882689-414e8666-aded-45a8-994f-141d295eacd4.jpg)


![Screenshot 06-02-2022 122003](https://user-images.githubusercontent.com/81313873/152678397-a1d7a201-915e-49eb-822b-eda37c63a504.jpg)


The prediction curve looks good, even if they look almost alike, we need to keep in mind that we're only predicting one day based on 60 days. So we need to feed it 60 days data in order for us to be able to predict the 61st day. If we're to predict the 62nd day, we would need to feed it 59 days data and train the model.

## Explaining Bitcoin 30 days Price Prediction

If we want to predict actual days in the future not just looking at the past performance or looking at the next day, we need to look at the actual prediction for that particular day in the future. This means we're looking at 60 days, and we're still just predicting one single value in the future. We're not going to predict a sequence of movements, we are going to predict one price in the future, in this case, we are looking at 30 days prediction.

We can do that by adding the future day in the prepare data by saying future_day is 30 days. Which means we're not predicting the 61st day but the 30th day after those 60 days.

![Screenshot 07-02-2022 231010](https://user-images.githubusercontent.com/81313873/152880843-60f9bf01-ef81-4fc6-9882-32398595bab4.jpg)

This is just a slight change that we can use to predict the future day, hence, 30 days after that. Once the model is trained, we can see the 30 days prediction into the future which looks actually better as we can see what the price would be in 30 days time. Interpreting the curve, bitcoin price would rise to 48k or more from its current price.

![Screenshot 07-02-2022 234255](https://user-images.githubusercontent.com/81313873/152885095-babff09a-7e83-4202-8d8f-471223da9609.jpg)
