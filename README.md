# Temitope's Portfolio

# PROJECT 1: Bitcoin Price Predictive Model
Date : 5/2/2022

## Why Price Prediction on Bitcoin?
Carrying out a price predictive model on Bitcoin with its volatility's nature, it is important to predict its next move with the data available and check the models' accuracy.

## Explaining Bitcoin Present Price Prediction
The relevant libraries were imported and loaded the data from yahoo finance API with the starting date from 2018,1,1 and ending date to be now -the current day- which is the time frame for the training data. 


![Screenshot 07-02-2022 231842](https://user-images.githubusercontent.com/81313873/152881811-850291aa-e4b7-42a1-8eac-e4d24dcea272.jpg)


Later, I prepared the data for the Neural Network and scaled-down the data within 0 and 1 so that the Neural Network would work better with it.

Next, we are going to choose a number for the prediction days. The prediction days will be the number of days we are basing our prediction on. We look at the number of past x_days e.g 60 days, and then we predict one day in the future, hence, we look at 60 days then we predict the 61st day.
Then we need to prepare the training data. We need to have x_data and y_data where y_data is going to be the result of the prediction. This is Supervised Learning, we are saying we want the x_data to be the 60 days (actual value) data which is more concise, and the y_data to be the result of the prediction.

![Screenshot 07-02-2022 232316](https://user-images.githubusercontent.com/81313873/152882297-76812fbe-2874-4aaf-9cad-cbf31942480d.jpg)



Next, we are going to build a Neural Network which is going to be the model that we are going to use for the prediction. We are going to equate the model to Sequential() and add LSTM layers (Long Short-Term Memory) and dropout layers. LSTM layers are going to be recurrent layers which we are going to use to memorise data since we are dealing with sequential data which has day 1, day 2, day 3, and so on. Those layers are powerful because they are specialised in that sort of data and LSTM is memorising the crucial information by feeding the data back into the Neural Network, while the dropout layer is to prevent overfitting. Then we're going to compile them all and train the model with 25 epochs.

![Screenshot 07-02-2022 232436](https://user-images.githubusercontent.com/81313873/152882464-e0aa2367-dab5-4961-884b-e90edf7ed29c.jpg)

Next, we are going to test the model. For this, we need to specify a time frame for the testing data, load some testing data, and then compare the test results, the prediction result on the test data with the actual result. Now we're going to load data from yahoo finance API but with a new date for the test_start but the same date for the end (end of the prediction which is the current date 'now'). After, we're going to plot both the actual price and the predicted price in the matplotlib with different colors so we can differentiate the prices and give it a title.  

![Screenshot 07-02-2022 232606](https://user-images.githubusercontent.com/81313873/152882689-414e8666-aded-45a8-994f-141d295eacd4.jpg)


![Screenshot 06-02-2022 122003](https://user-images.githubusercontent.com/81313873/152678397-a1d7a201-915e-49eb-822b-eda37c63a504.jpg)


The prediction curve looks good, even if they look almost alike, we need to keep in mind that we are only predicting one day based on 60 days. So we need to feed it 60 days of data for us to be able to predict the 61st day. If we are to predict the 62nd day, we would need to feed it 59 days of data and train the model.

## Explaining Bitcoin at a 30 day Price Prediction

If we want to predict actual days in the future not just looking at the past performance or looking at the next day, we need to look at the actual prediction for that particular day in the future. This means we are looking at 60 days, and we are still just predicting one single value in the future. We are not going to predict a sequence of movements, we are going to predict one price in the future, in this case, we are looking at 30 days prediction.

We can do that by adding the future day in the prepared data by saying future_day is 30 days. This means we are not predicting the 61st day but the 30th day after those 60 days.

![Screenshot 07-02-2022 231010](https://user-images.githubusercontent.com/81313873/152880843-60f9bf01-ef81-4fc6-9882-32398595bab4.jpg)

This is just a slight change that we can use to predict the future day, hence, 30 days after that. Once the model is trained, we can see the 30 days prediction into the future. The forcast is positive as we can see what the price would be in 30 days. Interpreting the curve, the bitcoin price would rise to $48k or more from its current price.

![Screenshot 07-02-2022 234255](https://user-images.githubusercontent.com/81313873/152885095-babff09a-7e83-4202-8d8f-471223da9609.jpg)

---

# PROJECT 2: Data Analysis for Predicting Wine Quality With Machine Learning

### Overview
A python project I worked on with a case study that predicts wine quality, in this case, red wine. I am working on this project using python on Jupyter notebook.

### Goal

Provide insight and predict the quality of the wine based on the features of the wine using a classification model.

### Analysis Approach

This dataset can either be worked on as a classification or a regression model. I chose to work with classification model so as to get more accuracy on the wine quality.

### Dataset Information
This dataset is gotten from UCI https://archive.ics.uci.edu/ml/datasets/Wine+Quality machine learning repository. The classes are ordered and not balanced that means a high number of samples are focused on only one class. The dataset needs to be balanced so as to get the accuracy correctly. Poor and excellent wines will be spotted by the few outliers.


Each wine sample has the following characteristics:
1. Fixed acidity
2. Volatile acidity
3. Citric sugar
4. Residual sugar
5. Chlorides
6. Free sulfur dioxide
7. Total sulfur dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol
12. Quality (values between 0 and 10)

### Modules Used

![Modules imported](https://user-images.githubusercontent.com/81313873/157775740-9fc46620-7112-4140-a85e-87220db6da0e.jpg)

### Loading the dataset

Loading the dataset in order to see all the 12 attributes of the data and making a copy of the data in case I want to make use of some attributes on the data without affecting the original data

![Dataset](https://user-images.githubusercontent.com/81313873/157825388-7d6f1602-43c9-4bb8-80de-c7f930a6e8ef.jpg)

I checked the statistical information to see if there were any missing values from the count. The total number of sample is 1599 and there are no missing values from the dataset. If there were missing values, they can be filled with the mean, median, or mode values but in numerical attributes, the mean is used to fill the missing values because of its average value.

![Dataset described](https://user-images.githubusercontent.com/81313873/157827432-22117524-bea7-4c2a-9dd6-3be6b4d8b6a2.jpg)

Checking the datatype info, we will see the different datatypes, in this case, we have both the float and int. We can make use of the int as a classifier or regressor because it is within a particular range of values. We can perform various analysis because of these numerical attributes (float, int).

![Data type](https://user-images.githubusercontent.com/81313873/157828940-a9bf5372-3d01-411b-91f6-e6ac676d71b4.jpg)

### Data Preprocessing

Preprocessing the data to check for null values. The dataset has no missing values and as mentioned earlier, if there were any missing values, it can be filled with the mean of the column of the missing values.

![Data preprocessing](https://user-images.githubusercontent.com/81313873/157832100-5edde07b-874e-4e1f-bd61-a91f3e74d503.jpg)

### Exploratory Data Analysis

Creating box plots to check for outliers

![Detecting outliers](https://user-images.githubusercontent.com/81313873/159095936-ddab4333-c521-48cc-90aa-64730cefff6f.jpg)

Running the code, there are big outliers that are in volatile acidity, chlorides, and total sulfur dioxide. By removing the outliers, we can improve the model accuracy by a few percent but it won't make that much difference. So, if we want to remove values from the outliers, it will be from volatile acidity, chlorides, and total sulfur dioxide, other values are in good range.

To check for the distribution and skewness of the values, we need to create a  distplot. Creating a distplot, all the values are in almost normal distribution so we do not need to change any of the distributions and the values range are very less except for free sulfur dioxide that is in a big range. To change the big range of free sulfur dioxide, we will use log transformation.

![Log transformation](https://user-images.githubusercontent.com/81313873/159507276-3194dfaf-3f2e-4c18-be30-5e7aa7c287f1.jpg)

Now, it is in uniform distribution unlike before when it was right skewed.

Carrying out another analysis to check the amount of class we have for the classification part.

![Amount of class](https://user-images.githubusercontent.com/81313873/159509124-e815b353-0e74-4694-afc3-9574e1934441.jpg)
As mentioned above, the quality of the wine is in the range of 0 to 10 but in this dataset, it is in the range of 3 to 9. Classes 5, 6, and 7 have higher number of samples compared to others. In this case, the whole model will be biased to these three classes only. We need to seperate the model for the test-dataset prediction which is called **Class Imbalancement** because all the classes are in different range of values. We will need to balance all the classes and make the dataset to be uniform.

### Correlation Matrix

![Correlation Matrix](https://user-images.githubusercontent.com/81313873/159523787-5a6bd535-2cd7-4abb-afdd-7b06bc339079.jpg)
Focusing on the output, alcohol is positively correlated with quality and negatively correlated with density while free sulfur dioxide is highly correlated with total sulfur dioxide because they are both sulfur dioxide. We can decide to drop free sulfur dioxide and also density from the dataset. Apart from these two values, we are not seeing any other highly imparted values in the correlation matrix.

### Class imbalacement
When it comes to class imbalancement, I used **SMOTE** and transformed the dataset. But first I checked the number of values in each dataset and 5 has the highest number of values of 681. The oversample function will sample all the low numbered classes to 681.

![Class imbalancement](https://user-images.githubusercontent.com/81313873/159694662-08d4abc4-9a8c-4718-b359-7caad0c9c7fb.jpg)

### Model Training
As I said above, I used classification for training the dataset. After clasifying the function, I splitted the dataset with the train_test_split function. After splitting the model, I trained the model and printed the accuracy. I used cross validation to be able to get more information about the performance of the model and multiplied by 100 so it will be in percentage format.

![Model training](https://user-images.githubusercontent.com/81313873/159907596-c14e8a3a-2555-47bd-be5e-7717766217b4.jpg)

To get the accuracy, I imported LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(), ExtraTreesClassifier(), lightgbm.LGBMClassifier(), and xgb.XGBClassifier(). From all the models, we have 86% accuracy and cross validation score of 81 from XGBClassifier which is one of the best models we have got from the dataset. If we did not rebalance the class or oversample the dataset, the percentage accuracy and the cross validation score would be low which shows that the model is not performing well.


To get the source code to the model, please **[click here](https://github.com/Temitope-Portfolio/Temitope-Portfolio/blob/main/Wine%20Quality.ipynb).**

---

# Project 3: Customer Shopping Analysis

### Some Information About The Dataset
The dataset recorded 99,457 customers from 10 different shopping malls in the city of Instabul.
