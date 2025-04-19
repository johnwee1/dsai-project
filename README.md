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




20% for the use of machine learning techniques to solve specific problem
20% for the presentation of data-driven insights and the recommendations
10% for the quality of your final team presentation and overall impressions
10% for learning something new and doing something beyond this course