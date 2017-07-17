# Fractal_Forecasting_Challenge
This project is to forecast price and sale volume of a set of items using LSTM model
## Problem Statement
Welcome to Antallagma - a digital exchange for trading goods. Antallagma started its operations 5 years back and has 
supported more than a million transactions till date. The Antallagma platform enables working of a traditional exchange 
on an online portal. 

On one hand, buyers make a bid at the value they are willing to buy ("bid value”) and the quantity they are willing to buy. 
Sellers on the other hand, ask for an ask price and the quantity they are willing to sell. The portal matches the buyers and 
sellers in realtime to create trades. All trades are settled at the end of the day at the median price of all agreed trades. 

You are one of the traders on the exchange and can supply all the material being traded on the exchange. 
In order to improve your logistics, you want to predict the median trade prices and volumes for all the trades 
happening (at item level) on the exchange. You can then plan to then use these predictions to create an optimized inventory strategy. 

We are expected to create trade forecasts for all items being traded on Antallagma along with the trade prices for a period of 6 months. 

## Solution
We will show the solution for one item from the entire dataset. The methods can be replicated for all other items.

As we can infer from the problem statement and the dataset(check my dataset folder) this problem belongs to forecasting price and the volume of the item sold. 
So for solving this challange, we can go for traditional but efficient *ARMA model*, which could capture the trend or any seosonality in the trading goods price and volume of the item sold. But considering the simplicity as well as the efficiency of the *Recurrent Neural Network (RNN)*. I have decided to go for this state of art approach method.

Let us first visualize the trend of the price variation and volume of item sold for a single item.
![price_treng](https://user-images.githubusercontent.com/14236684/28273039-33be03ea-6adb-11e7-86ac-9acdbc2b8108.PNG)
![volume_trend](https://user-images.githubusercontent.com/14236684/28273125-7644d298-6adb-11e7-8af6-cdb714c8a417.PNG)

Now the challenge is to model an algorithm which could study the above two trend and try to fit a line which capture the variation of these two values.

Let us see how RNN can come to a rescue and you will be amazed (so was I) that how powerful this model is to learn the trend from the input it has received.

### Note:
I have only study the trend for only single item from the entire dataset, but the entire logic can be extended for other items too without any major in the logic. Please refer to my code for the entire logic.

### Hperparameter used
look_back = 7 days. <br />
optimizer (gradient function) = 'adam' <br />
loss function = 'mean_squared_error' 
batch_size = 32

## Result

loss: 1.5982e-04 <br />
Train Score: 47.60189 RMSE












