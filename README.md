# Earning Manipulation Detection Using Decision Tree, Random Forest, and Logistic Regression
# Problem definition
A bank provides loans to small and medium-sized businesses, ranging from 10 million to 500 million rupees. The bank suspects that some of its customers might engage in earnings manipulation to increase their chances of securing a loan. Therefore, the bank aims to identify earnings manipulation by its customers.

Banks and investment companies pay special attention to business earnings, which increases the motivation for business management to manipulate earnings to attract more capital in the market. Banks are aware of this and use models to examine business accounts and detect accounting fraud. Earnings manipulation occurs when management violates generally accepted accounting principles to show profitable financial performance. A famous example is Enron, which managed to appear successful for years through earnings manipulation until this accounting fraud was uncovered, leading to its bankruptcy in 2001.

In accounting, there is a well-known model called the Beneish model, introduced in 1999, which evaluates companies based on eight accounting criteria. After calculating these indices, the calculated score, known as the M-Score, is obtained as follows:

Beneish M-Score= $−4.84+0.92⋅DSRI+0.528⋅GMI+0.404⋅AQI+0.892⋅SGI+0.115⋅DEPI−0.172⋅SGAI+4.679⋅TATA−0.327⋅LVGI$

If this score is greater than -1.78, the company is likely manipulating earnings. However, if the score is less than or equal to -1.78, the likelihood of earnings manipulation is low. Since this model was developed for the United States, its implementation in other countries may not be as effective.
 # Data
 The data set contains eight accounting criteria used in the Beneish model and whether the company has manipulated its earning or not for 1000 companies.
 # Investigation done and insights derived
1. Using the Beneish model as the baseline model. Determining its accuracy for the data in this case.
2. In this data set, the number of companies engaging in earnings manipulation constitutes a small percentage of the total companies, i.e., less than 4% of companies. Identifying modeling problems in a binary classification problem with such an imbalanced dataset, where one class is much smaller than the other. Providing a suggestion to address this issue.
3. Developing a predictive model to detect earnings manipulation, using logistic regression, and then evaluating the model.
4. Developing a predictive model to detect earnings manipulation, using the CART model, and then evaluating the model. Explaining the insights gained from using the CART model.
5. Develop a model using random forest to detect earnings manipulation.
6. Providing final recommendation to the bank for using the above models in detecting earnings manipulation.
