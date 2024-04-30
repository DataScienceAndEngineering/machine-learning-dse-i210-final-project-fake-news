# The Proposal of Fake News Detection

😸 Authors: Abhina Premachandran Bindu, Esther, Mahbuba, Zhongming



## ❤️ Motivation

Fake news detection is a binary classification problem where the goal is to distinguish between genuine news articles and fabricated or misleading content. This task holds significant importance in today's digital age where misinformation can spread rapidly through social media and online platforms, leading to harmful consequences such as public panic, erosion of trust in media sources, and manipulation of public opinion. 

The question about fake news detection is more of an application-oriented question rather than a theoretical one. It pertains to the practical implementation of machine learning algorithms and natural language processing techniques to address a real-world problem of identifying misleading or deceptive information circulated through various media channels



## 👉 Method

In the realm of fake news detection, the aim is to deploy a combination of machine learning(ML) techniques and natural language processing (NLP) methods to effectively discern between legitimate news articles and fabricated or misleading content. 

First,  we apply typical steps of NLP to preprocess the text data and extract relevant features. 
- **Text Preprocessing:** Tokenization, Normalization, Stopword Removal, Stemming, Text Cleaning(e.g. HTML tags, special characters, or URLs)
- **Feature Extraction:** Bag of Words, TF-IDF


Next, we try 4 supervised ML algorithms and observe their performances.

- **Logistic Regression:** It's a common choice due to its simplicity and interpretability, making it well-suited for binary classification tasks like distinguishing between real and fake news.
- **Support Vector Machines:** SVMs excel at separating data points into distinct classes by finding the optimal hyperplane that maximizes the margin between them. They can effectively handle high-dimensional feature spaces and nonlinear decision boundaries.
- **Random Forests:** These are versatile ensemble learning methods that can handle high-dimensional data and capture complex relationships between features, making them effective for identifying patterns indicative of fake news.
- **Gradient Boosting Machines:** GBM algorithms like XGBoost or LightGBM are powerful ensemble methods that sequentially build a series of weak learners, continuously improving the model's predictive performance. They often yield state-of-the-art results in various classification tasks, including fake news detection.
- *Naive Bayes* assumes independence between features, which might not hold true for the complex linguistic patterns present in news articles. *kNN* relies on similarity measures between data points and might struggle with high-dimensional data or require careful feature engineering to perform effectively. 




## 👍 Evaluation

In our approach, the model evaluation consists of 3 phases. 

**Initial Model Evaluation:**

- Start by splitting the dataset into training and validation/test sets
- Train models using default hyperparameters
- Evaluate the model's performance on the validation/test set using appropriate metrics (e.g., accuracy, precision, recall, F1-score).
- This initial evaluation gives a baseline performance measure to compare against when tuning hyperparameters.

**Hyperparameter Tuning:**
- Use GridSearchCV or RandomSearchCV to explore the hyperparameter space and find the optimal combination of hyperparameters.

**Final Model Evaluation:**
- Once we've tuned the hyperparameters, retrain the model using the entire training dataset (including the validation set, if applicable) with the optimized hyperparameters.
- Evaluate the final tuned model on a separate holdout test set that has not been used during training or hyperparameter tuning.
- This final evaluation provides an unbiased estimate of the model's performance on unseen data and helps assess its generalization ability.


📊 Citations

- [Dataset#1](https://github.com/several27/FakeNewsCorpus/releases/tag/v1.0)
- [Dataset#2](https://figshare.com/articles/dataset/Fake_and_True_News_Dataset/13325198)
- [Prior Research](https://www.geeksforgeeks.org/fake-news-detection-using-machine-learning)

