# NBA Game Predictor

### IMPORTANT NOTES FOR MARKERS TO RUN THE FILES:

Data source: https://www.kaggle.com/datasets/nathanlauga/nba-games

Make sure you change PATH_DIR at the top of the code to whatever your directory set-up is 

You can download and use our x.pkl and y.pkl files to save time, or alternatively, change load at the top of the code to false and wait around 10 minutes as the files are generated later in the code



## Introduction

The deep learning model we built for this project was a recurrent neural network (RNN) to predict the winner of NBA games given a sequence of the involved teams’ recent games histories, as well as general performance context. More specifically, it uses Long Short-Term Memory RNN within it. These types of RNNs are useful for handling long-term dependencies in sequential data. Using memory cells and gating, their recurrent connections select which inputs to remember or forget. This is what gives LSTM RNNs the ability to handle long-term dependencies better than a standard RNN.

The task given to the LSTM RNN we built is to predict the outcome of NBA games. This specific deep learning model is suitable for this task because the input data spans many years of data collection, with thousands of individual games. The input given to the model is a vector containing:

(1) a sequence of game data of the home team’s last 10 games 

(2) a sequence of game data of the away team’s last 10 games

(3) a sequence of game data of the two team’s last 3 matchups

(4) the home team’s seasonal performance records (total win %, home win %, away win %)

(5) the away team’s seasonal performance records (total win %, home win %, away win %)

The data comes from spreadsheets and is converted into a sequence represented as vectors.The model analyzes the data and outputs a prediction of which of the two teams will win in an immediate matchup between the teams. With the model’s ability to use long-term memory, it can use many hundreds or thousands of data entries to formulate a prediction. The task of predicting an NBA game is a binary classification, and the output is either 1, indicating the home team wins, or 0, indicating the away team wins.

## Model Figure

![image2](https://user-images.githubusercontent.com/74215622/233810272-7aa709d3-4c51-43cb-8dd8-a38f084ca52e.png)

## Model Parameters

The first two LSTMs take a sequence of 10 matrices of size 64 (batch size) x 16 (the number of columns we retain in game history data). Thus it takes a tensor of size 64 x 10 x 16 and we take from its output the final hidden state, a tensor of size 1 x  64 (batch size) x 64 (hidden units)

The third LSTM is identical to the first two except for the fact that it takes a sequence of 3 matrices, thus a tensor of 64 x 3 x 14 (number of columns we retain in matchup history), and an output of 1 x 64 x 64 as well.

All these 3 outputs are concatenated together to form one of size 64 x 192 (which is 64 x 3) and run through a series of linear layers that first outputs 64 x 128, and then 64 x 64.

Meanwhile, the home and away ranking data are each run through a series linear layers that takes in an input of size 64 (batch size) x 1 x 6 (numbers of columns we retain from rankings). The dimensions go through the layers as:

64 x 1 x 6 -> 64 x 1 x 48 -> 64 x 1 x 96 -> 64 x 1 x 48 -> 64 x 1 x 6

Then the two outputs are concatenated together into size 64 x 1 x 12

Then this is concatenated with the LSTMs output of 64 x 64 to form one of size 64 x 76 (64+12)
This is fed into one last linear layer to output a scalar, which is run through the sigmoid function to get a final binary prediction.

## Model Examples


Correctly predicted example, outputted 0 and the label was 0:

[tensor([[[1.6106e+09, 1.6106e+09, 2.0200e+03, 6.2500e-01, 4.8300e-01,
          8.2400e-01, 3.3300e-01, 4.8000e-01, 2.9630e-01, 9.1667e-01,
          6.4000e-01, 9.5200e-01, 5.8500e-01, 7.2000e-01, 5.8025e-01,
          0.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 6.3095e-01, 5.0600e-01,
          7.1400e-01, 2.5000e-01, 4.8000e-01, 5.3086e-01, 6.5476e-01,
          4.5300e-01, 9.0900e-01, 3.6400e-01, 5.8000e-01, 4.5679e-01,
          0.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 5.8929e-01, 4.2000e-01,
          8.0600e-01, 1.7100e-01, 3.2000e-01, 4.9383e-01, 6.6071e-01,
          5.5600e-01, 6.2500e-01, 3.9300e-01, 5.4000e-01, 4.6914e-01,
          0.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 5.8929e-01, 3.5700e-01,
          7.2000e-01, 3.0600e-01, 4.2000e-01, 7.0370e-01, 6.1310e-01,
          4.1100e-01, 8.1300e-01, 2.7900e-01, 5.0000e-01, 6.0494e-01,
          0.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 5.5357e-01, 4.3400e-01,
          7.0600e-01, 3.6000e-01, 5.2000e-01, 5.9259e-01, 6.1905e-01,
          4.5300e-01, 7.1400e-01, 2.5000e-01, 4.2000e-01, 5.9259e-01,
          0.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 6.1905e-01, 4.8800e-01,
          6.0000e-01, 4.0600e-01, 5.4000e-01, 4.3210e-01, 6.7262e-01,
          5.1700e-01, 8.9500e-01, 1.8200e-01, 4.0000e-01, 5.4321e-01,
          0.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 7.5000e-01, 5.2900e-01,
          8.3300e-01, 4.4400e-01, 6.0000e-01, 6.2963e-01, 5.8333e-01,
          3.9100e-01, 6.5500e-01, 2.9700e-01, 4.6000e-01, 4.6914e-01,
          1.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 7.2619e-01, 5.1200e-01,
          7.8900e-01, 4.8700e-01, 5.6000e-01, 4.9383e-01, 6.3095e-01,
          4.6400e-01, 6.7900e-01, 3.9100e-01, 4.4000e-01, 5.5556e-01,
          1.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 6.9048e-01, 5.0000e-01,
          8.3300e-01, 3.3300e-01, 6.0000e-01, 5.9259e-01, 6.5476e-01,
          4.3800e-01, 6.4300e-01, 4.1500e-01, 6.6000e-01, 4.3210e-01,
          1.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 6.3690e-01, 4.4000e-01,
          6.2900e-01, 3.3300e-01, 4.6000e-01, 6.0494e-01, 6.3095e-01,
          3.8500e-01, 8.1500e-01, 2.3800e-01, 4.2000e-01, 6.9136e-01,
          1.0000e+00]]]), tensor([[[1.6106e+09, 1.6106e+09, 2.0200e+03, 6.2500e-01, 4.8300e-01,
          8.2400e-01, 3.3300e-01, 4.8000e-01, 2.9630e-01, 9.1667e-01,
          6.4000e-01, 9.5200e-01, 5.8500e-01, 7.2000e-01, 5.8025e-01,
          0.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 7.2024e-01, 5.4500e-01,
          7.3700e-01, 3.3300e-01, 5.4000e-01, 5.4321e-01, 5.9524e-01,
          4.3800e-01, 8.8900e-01, 3.5900e-01, 4.4000e-01, 4.4444e-01,
          1.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 6.3095e-01, 4.7400e-01,
          8.3300e-01, 4.1500e-01, 4.0000e-01, 6.0494e-01, 6.0714e-01,
          4.4700e-01, 6.6700e-01, 3.6400e-01, 5.2000e-01, 4.8148e-01,
          1.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 6.5476e-01, 5.0000e-01,
          6.8800e-01, 3.0600e-01, 4.4000e-01, 6.1728e-01, 5.8929e-01,
          4.6700e-01, 1.0000e+00, 2.7800e-01, 5.0000e-01, 4.5679e-01,
          1.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 7.5000e-01, 5.5700e-01,
          7.1400e-01, 4.3900e-01, 4.8000e-01, 6.5432e-01, 5.5952e-01,
          4.0200e-01, 6.5000e-01, 3.0400e-01, 4.4000e-01, 4.3210e-01,
          1.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 7.5595e-01, 4.8300e-01,
          8.7000e-01, 4.5700e-01, 5.4000e-01, 4.6914e-01, 7.1429e-01,
          5.1800e-01, 8.5000e-01, 4.4700e-01, 6.2000e-01, 4.6914e-01,
          1.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 7.3810e-01, 5.2900e-01,
          9.0500e-01, 3.9500e-01, 5.6000e-01, 6.0494e-01, 6.9048e-01,
          4.3900e-01, 7.0600e-01, 3.5300e-01, 5.0000e-01, 4.9383e-01,
          1.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 7.0833e-01, 4.6600e-01,
          9.2000e-01, 3.3300e-01, 5.2000e-01, 5.0617e-01, 6.9048e-01,
          4.5100e-01, 8.7500e-01, 2.9500e-01, 2.8000e-01, 5.9259e-01,
          1.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 5.8333e-01, 4.1200e-01,
          7.6200e-01, 3.0000e-01, 3.0000e-01, 6.5432e-01, 6.2500e-01,
          4.4000e-01, 7.0600e-01, 4.0600e-01, 3.4000e-01, 4.8148e-01,
          0.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 5.5357e-01, 4.2900e-01,
          9.3800e-01, 3.1600e-01, 2.6000e-01, 5.5556e-01, 6.4881e-01,
          4.3000e-01, 7.6200e-01, 3.3300e-01, 4.6000e-01, 6.5432e-01,
          0.0000e+00]]]), tensor([[[2.0190e+03, 6.0119e-01, 4.8800e-01, 5.8300e-01, 3.8500e-01,
          4.0000e-01, 4.5679e-01, 7.3214e-01, 5.0600e-01, 9.5800e-01,
          3.3300e-01, 4.8000e-01, 5.8025e-01, 0.0000e+00],
         [2.0200e+03, 6.6667e-01, 4.7700e-01, 6.3600e-01, 3.3300e-01,
          4.6000e-01, 4.6914e-01, 7.6190e-01, 4.5500e-01, 8.5700e-01,
          3.6000e-01, 4.4000e-01, 6.4198e-01, 0.0000e+00],
         [2.0200e+03, 6.2500e-01, 4.8300e-01, 8.2400e-01, 3.3300e-01,
          4.8000e-01, 2.9630e-01, 9.1667e-01, 6.4000e-01, 9.5200e-01,
          5.8500e-01, 7.2000e-01, 5.8025e-01, 0.0000e+00]]]), tensor([[[72.0000, 31.0000, 41.0000,  0.4310,  0.4444,  0.4167]]],
       dtype=torch.float64), tensor([[[72.0000, 52.0000, 20.0000,  0.7220,  0.8611,  0.5833]]],
       dtype=torch.float64), tensor([0])]

Incorrectly predicted example, outputted 1 and the label was 0:

[tensor([[[1.6106e+09, 1.6106e+09, 2.0200e+03, 5.1786e-01, 3.6000e-01,
          6.3600e-01, 2.1400e-01, 4.0000e-01, 6.0494e-01, 6.7262e-01,
          4.6400e-01, 7.0800e-01, 3.9100e-01, 6.4000e-01, 6.1728e-01,
          0.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 6.4286e-01, 4.3300e-01,
          7.8100e-01, 2.0000e-01, 4.0000e-01, 5.8025e-01, 7.3214e-01,
          5.3400e-01, 8.5700e-01, 3.8600e-01, 6.2000e-01, 4.6914e-01,
          0.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 6.4286e-01, 4.0200e-01,
          8.1800e-01, 4.0000e-01, 5.6000e-01, 7.0370e-01, 6.1310e-01,
          4.1900e-01, 7.1400e-01, 3.7700e-01, 4.4000e-01, 5.9259e-01,
          1.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 7.0238e-01, 4.8800e-01,
          8.3300e-01, 4.0400e-01, 7.2000e-01, 6.0494e-01, 5.7738e-01,
          4.0900e-01, 8.0000e-01, 2.3100e-01, 4.2000e-01, 4.8148e-01,
          1.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 8.0952e-01, 5.8000e-01,
          8.3300e-01, 5.0000e-01, 6.2000e-01, 5.6790e-01, 5.7738e-01,
          4.0400e-01, 6.0000e-01, 2.4300e-01, 4.2000e-01, 5.0617e-01,
          1.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 7.0833e-01, 4.6600e-01,
          9.2000e-01, 3.3300e-01, 5.2000e-01, 5.0617e-01, 6.9048e-01,
          4.5100e-01, 8.7500e-01, 2.9500e-01, 2.8000e-01, 5.9259e-01,
          1.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 7.2619e-01, 4.9500e-01,
          8.5000e-01, 2.7500e-01, 6.6000e-01, 4.9383e-01, 6.9048e-01,
          5.1900e-01, 9.0000e-01, 4.3200e-01, 5.2000e-01, 4.4444e-01,
          1.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 7.4405e-01, 4.8300e-01,
          8.0000e-01, 4.3600e-01, 5.8000e-01, 5.6790e-01, 7.2619e-01,
          4.6500e-01, 6.7900e-01, 4.0700e-01, 4.2000e-01, 6.2963e-01,
          1.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 6.7262e-01, 4.9400e-01,
          8.8900e-01, 3.8500e-01, 5.2000e-01, 5.6790e-01, 6.0119e-01,
          4.3500e-01, 7.8900e-01, 2.4000e-01, 5.0000e-01, 4.8148e-01,
          1.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 6.1310e-01, 4.0700e-01,
          7.6000e-01, 3.2300e-01, 5.4000e-01, 6.0494e-01, 5.9524e-01,
          4.4600e-01, 7.3300e-01, 4.4100e-01, 3.8000e-01, 5.6790e-01,
          1.0000e+00]]]), tensor([[[1.6106e+09, 1.6106e+09, 2.0200e+03, 6.1905e-01, 3.9500e-01,
          8.3300e-01, 3.9000e-01, 5.2000e-01, 5.6790e-01, 7.0238e-01,
          5.0600e-01, 7.7800e-01, 5.1900e-01, 3.8000e-01, 5.5556e-01,
          0.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 8.0357e-01, 5.2100e-01,
          8.1800e-01, 4.8700e-01, 6.4000e-01, 5.0617e-01, 8.2738e-01,
          5.3600e-01, 7.5000e-01, 4.8300e-01, 6.0000e-01, 5.6790e-01,
          0.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 6.6071e-01, 5.3000e-01,
          8.5700e-01, 3.9300e-01, 4.8000e-01, 4.8148e-01, 5.7738e-01,
          4.2700e-01, 6.4700e-01, 2.7000e-01, 5.0000e-01, 5.1852e-01,
          1.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 5.8929e-01, 3.8300e-01,
          7.0800e-01, 3.1300e-01, 3.6000e-01, 5.6790e-01, 6.4881e-01,
          4.4700e-01, 7.3100e-01, 3.2600e-01, 4.8000e-01, 6.9136e-01,
          0.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 6.8452e-01, 4.9500e-01,
          5.9300e-01, 3.0000e-01, 6.6000e-01, 6.0494e-01, 6.5476e-01,
          4.4300e-01, 6.6700e-01, 4.3500e-01, 5.8000e-01, 6.1728e-01,
          1.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 7.9167e-01, 5.7300e-01,
          9.3800e-01, 4.8500e-01, 6.4000e-01, 5.4321e-01, 6.1905e-01,
          4.2000e-01, 9.4700e-01, 3.1600e-01, 4.6000e-01, 4.3210e-01,
          1.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 6.9048e-01, 5.0000e-01,
          8.3300e-01, 3.3300e-01, 6.0000e-01, 5.9259e-01, 6.5476e-01,
          4.3800e-01, 6.4300e-01, 4.1500e-01, 6.6000e-01, 4.3210e-01,
          1.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 6.3690e-01, 4.4000e-01,
          6.2900e-01, 3.3300e-01, 4.6000e-01, 6.0494e-01, 6.3095e-01,
          3.8500e-01, 8.1500e-01, 2.3800e-01, 4.2000e-01, 6.9136e-01,
          1.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 6.7262e-01, 4.9400e-01,
          8.8900e-01, 3.8500e-01, 5.2000e-01, 5.6790e-01, 6.0119e-01,
          4.3500e-01, 7.8900e-01, 2.4000e-01, 5.0000e-01, 4.8148e-01,
          1.0000e+00],
         [1.6106e+09, 1.6106e+09, 2.0200e+03, 5.9524e-01, 4.4400e-01,
          5.4200e-01, 3.1800e-01, 3.6000e-01, 6.9136e-01, 5.7143e-01,
          3.5100e-01, 8.3300e-01, 3.6400e-01, 4.0000e-01, 6.7901e-01,
          1.0000e+00]]]), tensor([[[2.0200e+03, 6.9048e-01, 4.8400e-01, 7.2200e-01, 3.6600e-01,
          6.2000e-01, 5.3086e-01, 6.1310e-01, 4.6400e-01, 8.1000e-01,
          3.0800e-01, 4.8000e-01, 5.6790e-01, 1.0000e+00],
         [2.0200e+03, 6.1310e-01, 3.4400e-01, 8.2400e-01, 3.1700e-01,
          4.6000e-01, 5.6790e-01, 6.6071e-01, 3.8900e-01, 8.7100e-01,
          3.1300e-01, 5.0000e-01, 7.4074e-01, 0.0000e+00],
         [2.0200e+03, 6.7262e-01, 4.9400e-01, 8.8900e-01, 3.8500e-01,
          5.2000e-01, 5.6790e-01, 6.0119e-01, 4.3500e-01, 7.8900e-01,
          2.4000e-01, 5.0000e-01, 4.8148e-01, 1.0000e+00]]]), tensor([[[72.0000, 39.0000, 33.0000,  0.5420,  0.6944,  0.3889]]],
       dtype=torch.float64), tensor([[[72.0000, 38.0000, 34.0000,  0.5280,  0.5000,  0.5556]]],
       dtype=torch.float64), tensor([0])]

## Data Source

The initial data consisted of three CSV files downloaded from https://www.kaggle.com/datasets/nathanlauga/nba-games. These files were the games.csv, ranking.csv, and game_details.csv. The games.csv consists of the final results and stats of all NBA games after 2003. The ranking.csv consists of the season records of all NBA teams after 2003. The game_details.csv consists of the box score of each individual NBA game after 2003.


## Data Summary
Number of games won from the 2014-15 season onward by the away and home team respectively (6522 and 4898 respectively)

![image1](https://user-images.githubusercontent.com/74215622/233810266-3bbaf263-62aa-41ad-b5a8-4a68bf7626ca.png)

There were thus 11420 played in that span, with a home win % of about 57%. Looking at this, we should hope our model predicts with better than 57% accuracy, as we could just pick the home team each time instead.

According to some research*, the best NBA game predictor models, such as FiveThirtyEight, have an upper bound of accuracy at a little over 70%. We thus went into this project hoping for an accuracy of about 65%, as it seemed like a reasonable middle ground between elite models and just random home prediction.

https://digitalcommons.bryant.edu/cgi/viewcontent.cgi?article=1000&context=honors_data_science , https://blog.albertkuo.me/post/2022-01-21-how-good-is-fivethirtyeight-s-nba-prediction-model/ , https://fivethirtyeight.com/features/how-well-did-our-sports-predictions-hold-up-during-a-year-of-chaos/

There are 2664 different players in the game_details data and 668628 individual game performances. We had initially planned to include this data in our model but we realized through exploration that the scope of the file was much larger than we expected and given the fact that the dates of the performance is not directly provided, it would be difficult to parse for post 2014 data as we desired. Also given the lack of information on player availability and injury contexts and the sheer number of highly specific game performance statistics, we ran the risk of making the model too complex, inefficient and overfitting. Thus we opted to not include the game_details data in our model.

The maximum team total points, assists and rebounds in a game in our dataset is 168, 50, 81 respectively. It is interesting to note that all these events occurred post 2014, indicative of the hyper-offensive play style that began around then. This supports our eventual decision to remove the older data.

Games played by season from 2014-15 onwards:

![image5](https://user-images.githubusercontent.com/74215622/233810275-bb0d1e32-59f0-4d27-a257-27e5a426bd20.png)


## Data Transformation

The data was downloaded in the form of CSV files, so pandas’ DataFrame was used as the data structure.

Before any transformations on the dataset took place, we decided to only include data from the 2014-15 NBA season and onwards in the dataset given to the model. This was done because the more recent years are more representative of the current state and style of play in the NBA.

Once only the relevant data remained, our first step to get it into a usable form for our model was to normalize any numerical statistical values that were not given as percentages from the games.csv. This was points, assists, and rebounds for both home and away teams. This was done for both the recent games for each team, as well as recent matchups between two teams. After, the data entries were sorted by date of when the game was played, and only the most recent games, decided by hyper parameter tuning, were returned. From this, all non-relevant data, like team ID or game ID were dropped.

Then from ranking.csv, the home records and away records for each team were converted from strings in the form of ‘wins-losses’ to percentages. Similarly, any non-relevant data was dropped, and the remaining data was returned.

After the DataFrames for the teams’ recent games sequence, recent matchup sequence, and current rankings for a given game were filtered and sorted, they were all converted to numpy arrays. This was done through iterating through each game played, going in order of date played. The result of each game was determined by whether the home team won, and this was stored in a numpy array to become the labels for the model training.

## Data Split

We chose a standard 60/20/20 data split, but chronologically. We felt that training the model from roughly the 2014-15 to the 2018-18 seasons (first 60% of games) would help it get acclimated to the model style of gameplay, we validated on the next 20% of chronological games and finally tested on the most recent 20% of game data. We wanted to see if we could accurately predict immediate contemporary games (for example a game tonight) by training through historical data, so this split was ideal for us. We did not want to train it on simply the most recent data to avoid overfitting to the most recent games. Initially we wanted to do a 30/10/10/30/10/10 split for training,validation, testing, training, validation, testing, but we found it difficult to train on disjoint sets and ultimately decided an directly chronological split was better.

## Training Curve 

![image3](https://user-images.githubusercontent.com/74215622/233810273-74217be3-7647-48d7-8ab9-c149aed5f280.png)

## Hyperparameter Tuning 

We chose our hyperparameter through a combination of trial and error as well as calculated decision. We trained our model on the training set and then we evaluated how well it performed on the validation set for each set of hyperparameters. For each hyperparameter, we did multiple tunings to find the optimal value. For example, our learning rate was chosen to be the low value of 0.0005 to avoid oscillation and divergence in the training curve we noticed from higher values. We also chose the dropout rate of p = 0.1 as we felt it was high enough to help avoid overfitting but low enough as to not damage our model’s performance, especially with the multiple uses of it. The same was done for our batch size and number of epochs, a decision made via trial and error and in a desire to avoid oscillation and overfitting.


## Quantitative Measures

For our model, testing accuracy and testing loss are very simple and straightforward ways to gauge the performance of our model. Since predicting the outcome of an NBA game is a binary classification, counting the number of correct predictions over total predictions made is a clear way to judge the model’s effectiveness. We used Binary Cross Entropy Loss as our loss function, as it is good for binary classification problems.

## Quantitative and Qualitative Results 

![image4](https://user-images.githubusercontent.com/74215622/233810274-ef0a62ef-498b-4f20-a1cd-7ab314e698d6.png)

![image3](https://user-images.githubusercontent.com/74215622/233810273-74217be3-7647-48d7-8ab9-c149aed5f280.png)

Final Training Accuracy: 74.64162135442413 in Epoch: 200  
Final Validation Accuracy: 72.66435986159169 in Epoch: 200  
Best Training Accuracy: 75.31718569780853 in Epoch: 188  
Best Validation Accuracy: 73.35640138408304 in Epoch: 150  

We took our best validation weights from Epoch 150 and used it to model on the test set, and came out with:  
Final Test Accuracy: 74.0484429065744  
Final Test Loss: 0.008039019305813272  

That is a 74% test accuracy, a figure we are very proud of given the low bar of 57%, our hope for around 65% and the fact that the best models in the world perform around that standard (of course there are some important caveats to that performance that we will discuss below). Our final test loss is also very low at 0.008.

In terms of qualitative results, there isn’t much to gauge as we are purely predicting a binary outcome. Our Learning curves appear relatively smooth and trending towards a convergence and our model trains relatively quickly in under 10 minutes, and outputs test predictions almost instantaneously.

## Justification of Results 

Our model performed better than we hoped, at predicting a winner given recent game history and team context. Still, it is a relatively inflexible model compared to the elite ones used today. 

For example, we have no context of player injury or availability, no details on game scheduling (is one team’s recent schedule more intense than the other’s, resulting in player fatigue?), no details on the difficulties of the recent team’s games or seasonal performance (does one team have a better record than it should because it has had an easier schedule or easier opponents?). 

There’s no information on individual player performances (seasonal or by game) that could influence future games (the “hot” or “cold” streak in sports), advanced statistics, or even things like player chemistry (has there been a feud or fight amongst teammates recently, is the current matchup a heat rivalry, etc) or game context (is this a must-win game for one team, but a blow-off game for the other?). 

So while our model performed well in this specific and coldly quantitative context, comparable to elite models, it is very unlikely to be able to compete head to head in a real world setting, where qualitative data is also very important to consider as well as more nuanced quantitative data. To put it bluntly, although our model performs better than we hoped, we still wouldn’t rely on it over more comprehensive existing models due to the narrowness of its scope. A model that takes 10 minutes to train is admirable from an efficiency standpoint, but not from a comprehensive one.

But given its success, we attribute this to our hypothesis of training on modern game data from the “small ball” era, starting from 2014 onwards. Our model isn’t extremely complicated and was designed to prevent overfitting to the earlier chronological data. We noticed an increase in accuracy for the validation set when the dropout layers were added. We stand by our decision to remove game_detail data containing box score information as although it may have widened the scope of our model, it may have decreased accuracy and increased model complexity and inefficiency. Our choice to use LSTMs on our sequential data paid off as we were able to find and store patterns on a sequence of previous games to predict a winner effectively.

## Ethical Consideration 
Similar to what we stated in our proposal, a game winner predicting model can be used for gambling purposes, which can have connections to organized crime and can contribute to destructive gambling addictions. But, even though our model is fairly accurate, it is still less accurate than standard existing and publicly available models, and we do not plan to use it for such purposes. We don’t plan on making this model public anyway. Outside of this consideration, there isn’t too much concern over its usage as it can’t be used to hurt or really even help anybody, and thus is immune to abuse. Its limitations and weaknesses are not prejudicial to anyone in a meaningful way. Our training data is open and publicly sourced, known and available, so there aren't any privacy concerns.

## Authors

Allen Chazhoor: Model Design, writing report, data exploration, training and tuning model, parameters and hyperparameters

Kevin Borja: Model Design, data transformation, writing report, model training and plotting results

Damon Bui: Model Design, data exploration, writing report, architecture diagram

