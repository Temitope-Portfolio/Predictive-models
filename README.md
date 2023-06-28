# Temitope's Portfolio

# PROJECT 1: Bitcoin Price Predictive Model
Date: 5/2/2022

## Why Price Prediction on Bitcoin?
Carrying out a price predictive model on Bitcoin with its volatility nature, it is important to predict its next move with the data available and check the models' accuracy.

## Explaining Bitcoin Present Price Prediction
The relevant libraries were imported and loaded the data from yahoo finance API with the starting date from 2018,1,1 and the ending date to be now -the current day- which is the time frame for the training data. 


![Screenshot 07-02-2022 231842](https://user-images.githubusercontent.com/81313873/152881811-850291aa-e4b7-42a1-8eac-e4d24dcea272.jpg)


Later, I prepared the data for the Neural Network and scaled down the data between 0 and 1 so that the Neural Network would work better with it.

Next, we are going to choose a number for the prediction days. The prediction days will be the number of days we are basing our prediction on. We look at the number of past x_days e.g. 60 days, and then we predict one day in the future, hence, we look at 60 days then we predict the 61st day.
Then we need to prepare the training data. We need to have x_data and y_data where y_data is going to be the result of the prediction. This is Supervised Learning, we are saying we want the x_data to be the 60 days (actual value) data which is more concise, and the y_data to be the result of the prediction.

![Screenshot 07-02-2022 232316](https://user-images.githubusercontent.com/81313873/152882297-76812fbe-2874-4aaf-9cad-cbf31942480d.jpg)



Next, we are going to build a Neural Network which is going to be the model that we are going to use for the prediction. We are going to equate the model to Sequential() and add LSTM layers (Long Short-Term Memory) and dropout layers. LSTM layers are going to be recurrent layers which we are going to use to memorize data since we are dealing with sequential data which has day 1, day 2, day 3, and so on. Those layers are powerful because they are specialized in that sort of data and LSTM is memorizing the crucial information by feeding the data back into the Neural Network, while the dropout layer is to prevent overfitting. Then we're going to compile them all and train the model with 25 epochs.

![Screenshot 07-02-2022 232436](https://user-images.githubusercontent.com/81313873/152882464-e0aa2367-dab5-4961-884b-e90edf7ed29c.jpg)

Next, we are going to test the model. For this, we need to specify a time frame for the testing data, load some testing data, and then compare the test results, the prediction result on the test data with the actual result. Now we're going to load data from yahoo finance API but with a new date for the test_start but the same date for the end (end of the prediction which is the current date 'now'). After, we're going to plot both the actual price and the predicted price in the matplotlib with different colors so we can differentiate the prices and give it a title.  

![Screenshot 07-02-2022 232606](https://user-images.githubusercontent.com/81313873/152882689-414e8666-aded-45a8-994f-141d295eacd4.jpg)


![Screenshot 06-02-2022 122003](https://user-images.githubusercontent.com/81313873/152678397-a1d7a201-915e-49eb-822b-eda37c63a504.jpg)


The prediction curve looks good, even if they look almost alike, we need to keep in mind that we are only predicting one day based on 60 days. So we need to feed it 60 days of data for us to be able to predict the 61st day. If we are to predict the 62nd day, we would need to feed it 59 days of data and train the model.

## Explaining Bitcoin at a 30 day Price Prediction

If we want to predict actual days in the future not just looking at the past performance or looking at the next day, we need to look at the actual prediction for that particular day in the future. This means we are looking at 60 days, and we are still just predicting one single value in the future. We are not going to predict a sequence of movements, we are going to predict one price in the future, in this case, we are looking at 30 days prediction.

We can do that by adding the future day in the prepared data by saying future_day is 30 days. This means we are not predicting the 61st day but the 30th day after those 60 days.

![Screenshot 07-02-2022 231010](https://user-images.githubusercontent.com/81313873/152880843-60f9bf01-ef81-4fc6-9882-32398595bab4.jpg)

This is just a slight change that we can use to predict the future day, hence, 30 days after that. Once the model is trained, we can see the 30 days prediction into the future. The forecast is positive as we can see what the price would be in 30 days. Interpreting the curve, the Bitcoin price would rise to $48k or more from its current price.

![Screenshot 07-02-2022 234255](https://user-images.githubusercontent.com/81313873/152885095-babff09a-7e83-4202-8d8f-471223da9609.jpg)

---

# PROJECT 2: Data Analysis for Predicting Wine Quality With Machine Learning

### Overview
A Python project I worked on with a case study that predicts wine quality, in this case, red wine. I am working on this project using Python on Jupyter Notebook.

### Goal

Provide insight and predict the quality of the wine based on the features of the wine using a classification model. 

### Analysis Approach

This dataset can either be worked on as a classification or a regression model. I chose to work with classification model so as to get more accuracy on the wine quality.

### Dataset Information
This dataset is gotten from UCI https://archive.ics.uci.edu/ml/datasets/Wine+Quality machine learning repository. The classes are ordered and not balanced which means a high number of samples are focused on only one class. The dataset needs to be balanced so as to get the accuracy correctly. The few outliers will spot poor and excellent wines.


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

I checked the statistical information to see if there were any missing values from the count. The total number of samples is 1599 and there are no missing values from the dataset. If there were missing values, they can be filled with the mean, median, or mode values but in numerical attributes, the mean is used to fill the missing values because of its average value.

![Dataset described](https://user-images.githubusercontent.com/81313873/157827432-22117524-bea7-4c2a-9dd6-3be6b4d8b6a2.jpg)

Checking the datatype info, we will see the different datatypes, in this case, we have both the float and int. We can make use of the int as a classifier or regressor because it is within a particular range of values. We can perform various analyses because of these numerical attributes (float, int).

![Data type](https://user-images.githubusercontent.com/81313873/157828940-a9bf5372-3d01-411b-91f6-e6ac676d71b4.jpg)

### Data Preprocessing

Preprocessing the data to check for null values. The dataset has no missing values and as mentioned earlier, if there were any missing values, it can be filled with the mean of the column of the missing values.

![Data preprocessing](https://user-images.githubusercontent.com/81313873/157832100-5edde07b-874e-4e1f-bd61-a91f3e74d503.jpg)

### Exploratory Data Analysis

Creating box plots to check for outliers

![Detecting outliers](https://user-images.githubusercontent.com/81313873/159095936-ddab4333-c521-48cc-90aa-64730cefff6f.jpg)

Running the code, there are big outliers that are in volatile acidity, chlorides, and total sulfur dioxide. By removing the outliers, we can improve the model accuracy by a few percent but it won't make that much difference. So, if we want to remove values from the outliers, it will be from volatile acidity, chlorides, and total sulfur dioxide, other values are in good range.

To check for the distribution and skewness of the values, we need to create a  distplot. Creating a distplot, all the values are in almost normal distribution so we do not need to change any of the distributions and the values range is very less except for free sulfur dioxide which is in a big range. To change the big range of free sulfur dioxide, we will use log transformation.

![Log transformation](https://user-images.githubusercontent.com/81313873/159507276-3194dfaf-3f2e-4c18-be30-5e7aa7c287f1.jpg)

Now, it is in uniform distribution unlike before when it was right skewed.

Carrying out another analysis to check the amount of classes we have for the classification part.

![Amount of class](https://user-images.githubusercontent.com/81313873/159509124-e815b353-0e74-4694-afc3-9574e1934441.jpg)
As mentioned above, the quality of the wine is in the range of 0 to 10 but in this dataset, it is in the range of 3 to 9. Classes 5, 6, and 7 have higher number of samples compared to others. In this case, the whole model will be biased to these three classes only. We need to separate the model for the test-dataset prediction which is called **Class Imbalancement** because all the classes are in different ranges of values. We will need to balance all the classes and make the dataset to be uniform.

### Correlation Matrix

![Correlation Matrix](https://user-images.githubusercontent.com/81313873/159523787-5a6bd535-2cd7-4abb-afdd-7b06bc339079.jpg)
Focusing on the output, alcohol is positively correlated with quality and negatively correlated with density while free sulfur dioxide is highly correlated with total sulfur dioxide because they are both sulfur dioxide. We can decide to drop free sulfur dioxide and also density from the dataset. Apart from these two values, we are not seeing any other highly imparted values in the correlation matrix.

### Class imbalacement
When it comes to class imbalancement, I used **SMOTE** and transformed the dataset. But first I checked the number of values in each dataset and 5 has the highest number of values of 681. The oversample function will sample all the low-numbered classes to 681.

![Class imbalancement](https://user-images.githubusercontent.com/81313873/159694662-08d4abc4-9a8c-4718-b359-7caad0c9c7fb.jpg)

### Model Training
As I said above, I used classification for training the dataset. After classifying the function, I splitted the dataset with the train_test_split function. After splitting the model, I trained the model and printed the accuracy. I used cross-validation to be able to get more information about the performance of the model and multiplied by 100 so it will be in a percentage format.

![Model training](https://user-images.githubusercontent.com/81313873/159907596-c14e8a3a-2555-47bd-be5e-7717766217b4.jpg)

To get the accuracy, I imported LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(), ExtraTreesClassifier(), lightgbm.LGBMClassifier(), and xgb.XGBClassifier(). From all the models, we have 86% accuracy and a cross-validation score of 81 from XGBClassifier which is one of the best models we have got from the dataset. If we did not rebalance the class or oversample the dataset, the percentage accuracy and the cross-validation score would be low which shows that the model is not performing well.


To get the source code to the model, please **[click here](https://github.com/Temitope-Portfolio/Temitope-Portfolio/blob/main/Wine%20Quality.ipynb).**

---

# Project 3: Customer Shopping Analysis
### Goal

Provide insight into customers' shopping experience, and be able to make data-driven recommendations and solutions about areas that are underperforming.

### Some Information About The Dataset
This dataset is gotten from **[Kaggle](https://www.kaggle.com/datasets/mehmettahiraslan/customer-shopping-dataset).** The dataset recorded 99,457 customers from 10 different shopping malls from 2021 to 2023 in the city of Istanbul. We have gathered data from various age groups and genders to provide a comprehensive view of shopping habits in Istanbul. The dataset includes essential information such as invoice numbers, customer IDs, age, gender, payment methods, product categories, quantity, price, order dates, and shopping mall locations.

### Content

Attribute Information:

-> invoice_no: Invoice number. Nominal. A combination of the letter 'I' and a 6-digit integer uniquely assigned to each operation.

-> customer_id: Customer number. Nominal. A combination of the letter 'C' and a 6-digit integer uniquely assigned to each operation.

-> gender: String variable of the customer's gender.

-> age: Positive Integer variable of the customers age.

-> category: String variable of the category of the purchased product.

-> quantity: The quantities of each product (item) per transaction. Numeric.

-> price: Unit price. Numeric. Product price per unit in Turkish Liras (TL).

-> payment_method: String variable of the payment method (cash, credit card, or debit card) used for the transaction.

-> invoice_date: The day when a transaction was generated.

-> shopping_mall: String variable of the name of the shopping mall where the transaction was made.

### Modules Used
I begin by importing the necessary Python packages and then loading the dataset into the Python environment.

![Modules imported](https://github.com/Temitope-Portfolio/Temitope-Portfolio/assets/81313873/192b2279-69a5-491b-a1f5-b8af01d1683a)

### Loading the dataset

Loading the dataset to see the 10 attributes of the data and also the first 5 and last 5 columns of the data.

![Dataset for customer shopping analysis](https://github.com/Temitope-Portfolio/Temitope-Portfolio/assets/81313873/ab959e06-d166-4bf8-b502-1a52fd76712a)

Checking the statistical information to see if there were any missing values from the data. The total number of samples is 99,457 and there are no missing values from the dataset.

![Dataset preprocessing](https://github.com/Temitope-Portfolio/Temitope-Portfolio/assets/81313873/d41d4518-2773-4b0e-945e-c3c35b3ea5d1)

### Exploratory Data Analysis

Among the 99,457 customers that visited the shopping malls, 60% are females while 40% are males.

![Percentage of Females to Males](https://github.com/Temitope-Portfolio/Temitope-Portfolio/assets/81313873/c1ca1dde-d852-4803-8c03-665774b23f74)

With such a gap in the gender customers, the females purchased the most.

![Gender purchase power](https://github.com/Temitope-Portfolio/Temitope-Portfolio/assets/81313873/a8af5d6b-5f94-4f72-ac8f-d1869233a9e4)

There were different ages of individual customers who purchased from the shopping malls, which made me group the ages into 7 groups that can be easily targeted while marketing. In this analysis, the age group that purchased and transacted the most was 35-44 while age groups of 25-34, 45-54, and 55-64 also had high spending power.

![Age distribution](https://github.com/Temitope-Portfolio/Temitope-Portfolio/assets/81313873/89e664db-52e7-4202-ba38-4fc201a99b16)

There were 8 categories of items sold in all the shopping malls. Clothing was purchased the most, while cosmetics, food & beverages came second and third, and books and souvenirs are the least purchased items in the category.

![Category purchased](https://github.com/Temitope-Portfolio/Temitope-Portfolio/assets/81313873/1f15e7cb-0966-48ef-ac79-3ad23175bf27)

Clothing, shoes, and technology brought in the most revenue among all other categories while books, food & beverages, and souvenirs brought in the least revenue.

![Category spending](https://github.com/Temitope-Portfolio/Temitope-Portfolio/assets/81313873/647a3d14-eee6-447d-87f6-01a4135584a6)

Carrying out another analysis to know which payment method is been used the most. From the analysis, cash was used most by both genders in all the shopping malls.

![Payment method](https://github.com/Temitope-Portfolio/Temitope-Portfolio/assets/81313873/0b73818f-cced-49fc-8691-8df9778716d2)

As mentioned above, cash is the most preferred method of payment, I went further to check how each gender relates to the different methods of payment. Using the barplot to check the payment method used by each gender, females used cash more as a payment method compared to males while males used both cards more compared to females.

![Gender and payment method](https://github.com/Temitope-Portfolio/Temitope-Portfolio/assets/81313873/137cd8de-decd-4473-b02a-b50015ea3d6d)

Checking the shopping malls with the number of transactions, Mall of Istanbul, Kanyon, and Metrocity have a higher number of customers visiting the shopping malls.

![Transactions per shopping malls](https://github.com/Temitope-Portfolio/Temitope-Portfolio/assets/81313873/6a9519d7-73e0-4149-981f-3d019bdb9086)

The shopping malls of mall of Istanbul, Kanyon, and Metrocity generated more sales compared to others.

![Revenue per shopping mall](https://github.com/Temitope-Portfolio/Temitope-Portfolio/assets/81313873/0c6b006f-9fd9-4592-bb36-d6a4591d2903)

Checking the distribution of age groups among both genders to know which of the categories the consumers visit the most. This crosstab is just to further confirm the above to show which age group is more active in the shopping experience.

![age group and gender distribution](https://github.com/Temitope-Portfolio/Temitope-Portfolio/assets/81313873/1df59dc5-8286-4f41-84e6-1d900aace3ad)

#### Analysing Monthly, Yearly, and Quarterly Revenue

In this section, I worked on the invoice date provided in the dataset in order to split it and create new labels  into **day of the week**, **month**, and **year**. With these labels, I can know the month, year, and quarterly sales of the shopping malls.

I began by copying the original data before working on it, then I worked on the invoice date in order to split it into day, month, and year using the datetime package imported. I then went ahead and used the new data to get a year_month column which will be used to get the monthly sales of each year. Below is the outcome of the data wrangling.

![Screenshot 27-06-2023 135606](https://github.com/Temitope-Portfolio/Temitope-Portfolio/assets/81313873/6916d666-6a00-4578-a806-68e24cb5893d)

With the new data, I went ahead to check daily transactions in all the shopping malls

![Daily transactions](https://github.com/Temitope-Portfolio/Temitope-Portfolio/assets/81313873/30125349-f77a-4a39-b903-2e9c9df53490)

But because there is no huge difference with the bar plot, I went ahead to show the difference with a line chart which starts from Monday through Sunday. With that, I can see the fluctuations during the week. It shows a pattern of transactions which on Monday has its highest volume of transactions then later drops on Tuesday through Wednesday, a little bit of increase on Thursday but a huge increase on Friday, drops on Saturday and steady increase on Sunday till there's another increase on Monday and the pattern continues in that range.

![Line chart for daily transactions](https://github.com/Temitope-Portfolio/Temitope-Portfolio/assets/81313873/9b4dcca0-10db-4398-956e-e84472c4c20c)

In getting to know the monthly total sales per year, I added a new label **year_month** into the dataframe, taking into account the total price of each month. Checking the chart, the months of October record the highest sales while the months of July come second.

![Total amount spent monthly per year](https://github.com/Temitope-Portfolio/Temitope-Portfolio/assets/81313873/f6da46e2-c87d-46ea-b5fc-c4abdc68f30f)

Checking the year that makes the most sales, the year 2022 had the most sales. The year 2023 can not be used as we are currently in it and it has not concluded. The year 2023 has data only from January to the beginning of March.

![Yearly sales](https://github.com/Temitope-Portfolio/Temitope-Portfolio/assets/81313873/f9bb5a19-58ac-4981-8ae4-62319072cd32)

In order to get the quarterly sales for the years 2021 and 2022, I added a new label **quarter** into the dataframe. In getting to know the pattern of the quarterly sales, I used the same method I used in the daily transaction with the line chart to be able to see the clear movement of the quarterly sales. This shows that sales always drop in the first quarter of the year, while sales peak in the third quarter of the year.

![Quarterly sales](https://github.com/Temitope-Portfolio/Temitope-Portfolio/assets/81313873/ffbff5c2-41a5-402a-b786-a7083759df27)

To get the source code to this analysis, kindly **[click here](https://github.com/Temitope-Portfolio/Temitope-Portfolio/blob/main/Customer%20Shopping%20Analysis.ipynb).**

### **Conclusion**

From the dataset, I highlighted the fact that **60%** of the customers visiting the shopping malls were females which is a major reason for the high sales of the shopping malls.

I also give a detailed analysis of the age group that purchases the most. Customers in the age groups **25-34, 35-44, 45-54, and 55-64** drive up the sales of the shopping malls which brings in more revenue.

Among the 8 categories of items sold, **clothing, shoes, and technology** ranked in more sales, in other words, more profits.

The customers preferred using cash as their mode of payment as most of the items were paid for with cash. In an instance where either credit or debit cards were used, the males used them more than the females as a mode of payment, while the females used cash more than the males.

In the dataset, we have 10 shopping malls in different locations in the city of Istanbul, and out of the 10 malls, 5 drive in most of the sales with the Mall of Istanbul and Kanyon outperforming all the malls.

From the analysis, I noticed that customers purchase items more on Mondays and Fridays as there is always a customer spike on these days, and they purchase less on Wednesdays and Saturdays.

In all the shopping malls altogether, the months of October and July always record high sales amidst other months and the year 2022 generated more sales compared to 2021.  No decision can be made on the year 2023 as it is not concluded. More sales were made in the third quarter of 2022. There is an exponential increase in the sales generated while comparing the quarter sales of each year. To expand on this, the sales generated in the first quarter of 2022 were higher than the 2021 first-quarter sales and 2022 second-quarter sales were higher than 2021 second-quarter sales and so on.
