# SC1015 Mini Project: Predicting Credit Risk

### Problem definition based on a dataset

Our dataset is linked here: https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data

Spanning a total of 12 features combined with both numerical and categorical data, we found that it was suitable to formulate our problem in terms of a binary classification problem - whether or not a given borrower, given the features provided, will ultimately default on the loan that he is applying for.

Each example contains information on the loan terms, the person's relevant financial and employment and housing info, the type of grade of loan the loan would be classified as, and ultimately the `loan_status` - whether they defaulted on the loan or not.

### Data preparation and cleaning

Data preparation involved spanned two parts - first, there are some malformed data that can be unequivocally dropped without further analysis, like completely empty rows, or duplicate data. The second part of data cleaning required us to look at the data first, then trim out parts of the data that made no sense based on the given context. The threshold for what constitutes "anomalous data" was based on sensical values at our discretion (i.e. borrower age likely isn't above 80).

### Exploratory data analysis/visualization

EDA was achieved using a combination of box and whisker plots to visualize the distribution of all features. Loan status was also used as a means of separating the data.
Correlation matrix was graphed and pairwise relationships were analysed for anomalous data. Some data that looked off was then graphed against each other to examine their relationship more carefully, and then cleaned off.

Based off the correlation matrix:

- loan status (last row) has \*_highest correlation magnitude with loan_percent_income and interest rate_

- person_income row:
  The higher the income, the higher the loan amount might be
  Interest rates show almost no correlation to loan profile which is true since in practice since they are driven by mkt forces
  The higher the income, the lower the loan expressed as % of income.

- loan percentage income & person income:
  The higher the loan amount, the higher they were as % of income, i.e. people with lower income take loans that result in higher leverage (to buy big ticket items)

- employment_length:
  We further investigate person_age and person_employment_length, since common sense deduction is that age and credit history length should in practice have stronger correlations. Our discussion is detailed in the video.

## Feature Engineering/Exploratory Analysis using PCA

We use the sklearn library to conduct principal component analysis with the goal of removing noise and keeping only the most important features.

However, realising that the total explained variance of the top 3 principal components only account for less than 30% of the total variance in our dataset, we realised we could not do dimensionality reduction (PCA) as it would be highly lossy (we would lose important information about our dataset)

Inspired by the highly complex nature of our dataset (as it couldn't be simplified using PCA), we decided to use neural networks as a classifier to predict credit risk.

## Neural Networks

We used the PyTorch library to train our neural network.

We settled for using 3 hidden layers in our neural network and experimented with various hyperparameters/settings, as we tried more/less hidden layers but they didn't improve accuracy significantly.

In training our neural network, we also faced the issue of overfitting, where the model performed better on the train set compared to the test set, and we used the regularization technique of dropout to reduce overfitting by forcing the model to rely less on individual neurons, which forced the model to generalise and learn rather than just memorising.

We also manually experimented with various hyperparameters to obtain the optimal accuracy of ~93% on both the train and test set.
As 93% was near the limit of all the machine learning models, we suspect that the other 7% of data might be anomalous data, which all models perform poorly on due to their anomalous nature which means they don't follow the general trends/pattern of the data

20% for the use of machine learning techniques to solve specific problem
20% for the presentation of data-driven insights and the recommendations
10% for the quality of your final team presentation and overall impressions
10% for learning something new and doing something beyond this course
