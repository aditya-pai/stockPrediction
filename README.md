----------REAMDE---------------

**------Author: ADITYA D PAI-----**


This is my project on a machine learning based system for stock investment predictions.
The premise of this work is to determine whether or not to invest in a company based on its current statistics along with its historical statistics of over a decade.


The objective is to use machine learning to analyze public company (stocks) fundamentals (things like price/book ratio, P/E ratio, Debt/Equity ... etc), and then classify the stocks as either out-performers compared to the market or under-performers and to develop an investment strategy to determine ideal investments.

The stocks taken into consideration in this project are of the S&P500 index. Taking an overview, this project trains the algorithm on the historical company statistics from the S&P 500 index, and then predicts whether a stock will outperform or underperform the market based on their current statistics. And on the basis of this prediction, it generates a list of the ideal stocks to invest in, whilst considering minimizing the associated risk.

The two files contained in this repo:

1. get_invests.py

Main script which generates final list of investment suggestions based on strategy.

2. testing.py

Script used for the benchmarking and backtesting framework in order to choose optimum algorithm for the model

**License:**

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.