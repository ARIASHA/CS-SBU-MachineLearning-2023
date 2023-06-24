# CS-SBU-MachineLearning-2023
Computer Science Faculty of Shahid Beheshti University. Winter 2023
## 1st Assignment
In this part, you are going to work with the News Popularity Prediction https://archive.ics.uci.edu/dataset/332/online+news+popularity dataset.
You will implement a regression model using the Scikit-Learn package to predict the popularity of new articles (the number of times they will be shared online) based on about 60 features. 

You are expected:

Perform exploratory data analysis on the dataset.

Try Ridge and Lasso regression.

Use various scaling methods and report their effects.

Add polynomial features and report their effect.

Try using GridSearchCV with RandomizedSearchCV to tune your model’s hyperparameters. (Extra Point)

Apply the feature selection methods that you have implemented in the above sections.

Get familiar with and implement the following loss functions from scratch and utilize them with a Linear Regression model and discuss their effect on the performance of the model. (Extra Point)
Absolute Error
Epsilon-sensetive error
Huber

Implement batch gradient descent with early stopping for softmax regression from scratch. Use it on a classification task on the Penguins dataset.(Extra Point)

## 2nd Assignment
In this part, you are going to work with the Vehicle Insurance Claim Fraud Detection dataset.
https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection 

You will implement multiple classification models using the Scikit-Learn package to predict if a claim application is fraudulent or not, based on about 32 features. You are expected:
Perform exploratory data analysis on the dataset.

Try to tackle the problem using the following models :

Logistic Regression

SVM

Decision-Trees

Random Forest

Other classifiers: KNN, Naive Bayes, Ensemble models (Extra Point)


Implement a Bagging classifier from scratch. You can use sklearn for the base model. Test your model on the Penguins dataset. (Extra Point)

## 3rd Assignme 
You are going to work on the Supermarket dataset for predictive marketing. Your task is to use clustering algorithms to segment the customers into distinct groups based on their shopping behavior and demographics.
https://www.kaggle.com/datasets/hunter0007/ecommerce-dataset-for-predictive-marketing-2023

Explore and preprocess the dataset. This may involve handling categorical variables and normalizing or scaling numerical features and feature engineering.

Use K-means clustering to identify the optimal number of clusters. Experiment with different values of K and use metrics such as the elbow method and silhouette score to evaluate the performance of the clustering.
Visualize the clusters and analyze their characteristics. This may involve plotting the clusters in 2D or 3D using PCA or t-SNE.

Experiment with other clustering algorithms such as DBSCAN or hierarchical clustering, and compare their performance with K-means.

Try to reduce data dimensionality using PCA before training your model, use different numbers of components and report their effects. (Extra Point)

Interpret the results and provide insights to the store owners. What are the distinct customer segments that have been identified? How can the store owners use this information to improve their marketing strategy, product offerings, or customer experience? (Extra Point)

