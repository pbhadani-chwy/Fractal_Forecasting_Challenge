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
So for solving this challange, we can go for traditional but efficient ARMA model, which could capture the trend or any seosonality in the trading goods price and volume of the item sold. But considering the simplicity as well as the efficiency of the Recurrent Neural Network (RNN) I have decided to go for this state of art approach method.

