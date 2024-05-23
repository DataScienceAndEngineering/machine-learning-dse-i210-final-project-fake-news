# Machine Learning Approaches for Distinguishing Real from Fake: A Study on Fake News Classification with NLP Techniques

## Abstract  

Traditional methods of identifying and combating fake news are often insufficient due to the sheer volume and speed at which it is disseminated. Leveraging machine learning (ML) techniques offers a promising solution to this problem. ML models can be trained to detect patterns and features indicative of fake news, such as linguistic cues, source credibility, and dissemination patterns. By automating the detection process, ML can quickly and efficiently analyze vast amounts of data, far beyond the capacity of human fact-checkers.

![banner](figures/banner.png)

Our main objective is to identify a pair of high-accuracy vectorizer and classifier that can effectively recognize fake news. To achieve this, we compared popular libraries such as NLTK, spaCy, and Gensim for text preprocessing and vectorization, resulting in three different vectors. We then trained models using six different classifiers, creating a total of 18 combinations. From these, we selected the combination with the highest accuracy score as our output.

Our secondary objective involves using spaCy and Gensim for entity recognition and topic modeling during text preprocessing to perform basic semantic analysis. Additionally, considering the interpretability challenges of the Word2Vec model, we used the TF-IDF model as a baseline. We employed SHAP and LIME to interpret feature importance, identifying the key terms that significantly influence the classifier.

## Introduction

The terms "fake news" and "misinformation" have seen a massive uptick in use within the past couple of years. The phenomenon of false information being spread throughout many forms of media (particularly the Internet) has caused concern for its impacts on health and wellness safety, political distrust, and social divisiveness. This, coupled with a growing epidemic of decreasing attention spans highlights the need for a fast and accurate way to detect fake news in order to avoid it or mitigate it's spread.

The dataset, obtained from Kaggle, focuses on real and fake news and includes four features: news title, news content, news category, and creation date. The labels are "true" and "fake." The goal of our research project is to extract data features using NLP techniques and then combine them with ML to find an effective classifier, which we will validate using recent real and fake news.

To achieve this, we broke down the research project into core issues:

- Vectorization is a key challenge. The dataset consists of 40,000 rows, with each row's news content averaging 300 tokens. Extracting features from such a large dataset is problematic. The traditional models we are familiar with, such as BoW and TF-IDF, are only effective for small datasets. For the larger dataset, we used Word2Vec and Doc2Vec.

- Choosing the classifier. We used logistic regression as a baseline and experimented with Decision Tree, SVM, and ensemble algorithms like Random Forest and Gradient Boosting. Although we did not explore every possible model, we refrain from concluding that we used the best models. Nevertheless, the selected models achieved an accuracy of approximately 90%.

- Interpreting the results is challenging. Text analysis is notoriously difficult to interpret, especially after word embedding when all words are vectorized. Determining which words significantly impact the label is challenging. Therefore, we used SHAP and LIME methods to interpret the results.

Our team's main workflow is illustrated in the diagram below:

![workflow](figures/workflow.png)

## Background

There are two main approaches for fake news detection with machine learning, depending on the availability of labeled data:  

**Supervised Learning**: This is the most common approach. Here's how it works:  
- Data Preparation: Massive amounts of text data are collected from news articles, social media posts, etc. This data is then labeled as "real" or "fake" by human experts.
- Feature Engineering: The text data is analyzed to extract key features. This could involve things like word usage, sentiment analysis, presence of ALL CAPS, exclamation points, etc.
- Model Training: Machine learning algorithms like Support Vector Machines (SVMs) or Logistic Regression are trained on the labeled data. The model learns to identify patterns that differentiate real and fake news based on the extracted features.
- Detection: Once trained, the model can be used to analyze new, unseen content. It assigns a probability of whether the new content is real or fake news based on the patterns it learned.

**Unsupervised Learning**: This approach is useful when labeled data is scarce. Here's the gist: 
- Clustering Algorithms: The data is analyzed to identify inherent groupings (clusters) within the text. These clusters may represent different types of content, like factual news articles, political commentary, or rumor mills.
- Anomaly Detection: By understanding the characteristics of each cluster, the algorithm can flag content that falls outside the norm and might be suspicious, potentially fake news.
- Both approaches have their advantages and disadvantages. Supervised learning offers higher accuracy, but requires a lot of labeled data which can be expensive and time-consuming to create. Unsupervised learning doesn't need labeled data, but it might be less accurate and require human intervention to interpret the flagged content.

Since there are labels in our dataset, we will follow the supervised learning methodology. But we will also performance unsupervised learning, such as sentiment analysis and topic modeling to view our data from different angles. 


## Data

[Data Source](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets?resource=download) is from Kaggle. The dataset contains two types of articles: fake and real News. This dataset was collected from real world sources; the truthful articles were obtained by crawling articles from Reuters.com (News website). As for the fake news articles, they were collected from different sources. The fake news articles were collected from unreliable websites that were flagged by Politifact (a fact-checking organization in the USA) and Wikipedia. The dataset contains different types of articles on different topics, however, the majority of articles focus on political and World news topics.  
The dataset consists of two CSV files. The first file named “True.csv” contains more than 12,600 articles from reuter.com. The second file named “Fake.csv” contains more than 12,600 articles from different fake news outlet resources. Each article contains the following information: article title, text, type and the date the article was published on. To match the fake news data collected for kaggle.com, we focused mostly on collecting articles from 2016 to 2017. The data collected were cleaned and processed, however, the punctuations and mistakes that existed in the fake news were kept in the text.

Sample Data: 

| title                 | text                                             | subject      | date       | type    |
| --------------------- | ------------------------------------------------ | ------------ | ---------- | ------- |
| As U.S. budget...     | The head of a conservative Republican faction... | politicsNews | 12/31/2017 | TRUE    |
| Donald Trump Sends... | He had to give a shout out to his enemies...     | News         | 12/31/2017 | FAKE    |


After data preprocessing, we can observe some data features:
- In terms of the amount of labels, our dataset is a very balanced one.
- The content length for both true and fake news are 300 tokens on average.
- The null value takes up 1%, which should be dropped instead of imputation.
- The news fall into 6 categories.
- We will only take column 'text' and 'type' into the NLP. 

![EDA Visualization](figures/EDA-1.png)


## Methods

<p>
  Our main goal was to build the best Machine Learning classifier that can accurately classify the data, which consists of news documents as real or fake. The process was quite challenging as we are dealing with complex text data. Inorder to accomplish this goal, our tasks were mainly focused on Data cleaning and Exploration, Text preprocessing using various NLP libraries, Feature extraction and building word vectors, Model building, training, and testing, Model Evaluation and Model Explainability.
</p>
<p>
  After loading the data and initial analysis of the data using basic pandas dataframe inspection methods along with missingno library plots, we found that the data is clean with no null values as well as the classes are balance with 23481 Fake and 21417 Real news data. Therefore, we decided to start with Exploratory Data Analysis. But for this step, we need to clean and tokenzie the text documents using NLP libraries. We used nltk, Genism as well as Spacy libraries for text preprocessing and cleaning. Now, the data is cleaned and tokenized, we applied various visualization techniques to understand the distribution of data. To understand the distribution of most frequent words in both classes, we created a wordcloud along with Frequency bar charts of words. But the results indicated that almost all words occured equally in both classes. Therefore, we decided to move on with utilizing Spacy's entity recognition and part of speech tagging methods. The results were interesting in the fact that the Fake news data comprised of few interjections which included profanities which were absent in True news data. 
</p>

The plot below shows the top five words for each part of speech tag. These results are after preprocessing that removed stop words and various other commonly used symbols. As was already mentioned, fake news seems to have a lot more interjections (red) than true news, most likely as a tactic to get people's attention and spread further. Also, true news tends to use the word "said" a lot more often than fake news.

<img src="https://github.com/DataScienceAndEngineering/machine-learning-dse-i210-final-project-fake-news/blob/main/reports/figures/fixed_pos.png" width="800"/>

The image below is a bar plot of the top five most common words for each entity. In this context an entity refers an object that could be:

- DATE: Refers to specific dates or periods mentioned in the text.
- PERSON: Individual people, including fictional and real names.
- NORP: Nationalities or religious or political groups.
- FAC: Buildings, airports, highways, bridges, etc.
- ORG: Companies, agencies, institutions, etc.
- GPE: Countries, cities, states.
- LOC: Non-GPE locations, mountain ranges, bodies of water.
- PRODUCT: Objects, vehicles, foods, etc. (Not services.)
- EVENT: Named hurricanes, battles, wars, sports events, etc.
- WORK_OF_ART: Titles of books, songs, etc.
- LAW: Named documents made into laws.
 -LANGUAGE: Any named language.

Here we can see that fake news tends to mention people with a higher frequency compared to true news. In addition to that, true news mentions geo-political entities and people with the same frequency, while fake news tends to talk about people more than anything else.

<img src="https://github.com/DataScienceAndEngineering/machine-learning-dse-i210-final-project-fake-news/blob/main/reports/figures/fixed_entity.png" width="800"/>

**Word Embedding**   
We tried three vectorization models, including tf-idf, word2vec and doc2vec.   
	- tf-idf is highly interpretable and good for small dataset. we use sklearn api.   
	- Word2vec takes the word orders into consideration, measures the similarity of words and runs fast for large dataset due to lower dimensionality. We tried both spaCy and gensim api.  
	- Doc2vec takes the whole content into analysis, finds similarity between contents and fits the content into a fixed-length vector, especially good for topic modeling. We tried gensim api.  

However, unlike tf-idf, the word2vec and doc2vec are not as easy to interprete and evaluate. The accuracy is highly related to the quality of corpora. In order to remove the bias of corpora and run it faster, we make use of spaCy and gensim pre-trained models. These models are well trained with wide-topic and super large dataset.   
	- spacy/en_core_web_sm  
	- gensim/word2vec-google-news-300  
 
In this step, we performed feature extraction with different models and get a number of large matrice below as input for the machine learning models:  
	- LDA-matrix, generated by gensim word2vec  
 	- BoW-matrix, generated by spaCy plus nltk  
  	- W2V-matrix, generated by spaCy word2vec  
   	- D2V-matrix, generated by gensim doc2vec  

**Model Training**  
For this binary classification, we tried multiple models, from weak learners to ensemble methods. As a preliminary analysis tfidf vectors were used to fit, train and predict more than 10 different classifiers through a loop. Out of the algorithms tested for accuracy, it is found that Decision tree, Adaboost, Bagging, and xgb have the highest accuracy score of 0.992 for the sampled dataset. Knn performed the worst with a score of 0.763. Futher in the project, after realizing that the model performance also varies with respect to the word embedding inputs used, classifier models were also trained along with LDA, BoW, W2V, D2V matrices as input. There are expected 4*5 pairs. We looped over all combinations to find the best combined.   
	- Logistics Regression(lr). Set as baseline model.  
	- Decision Tree(dt).  
	- Support Vector Machine(svm).   
	- Random Forest(rf).  
	- Gradient Boosting(gb).  
 Out of all the combinations trained, we found that spaCy word2vec has the best result with SVM and RandomForest, and runs faster.

<img src="https://github.com/DataScienceAndEngineering/machine-learning-dse-i210-final-project-fake-news/blob/main/reports/figures/model_selection.jpg" width="800"/>

 **Model Explainability**
 <p>
	 we used three different methods of model explainability:
	 
1. plot_tree function of decision tree to visualize the performance of the model as well as to identify the important features. 

![DecionTree plot](figures/tree_plot_model_explainability.png)
	 The above tree visualization of the classifier indicates that the classifier uses 'said' feature as one of the main feature to decide whether the text is fake or real. In the next level, 'minist' and 'via' are used to split the data into the respective classes based on certain threshold values for the features.

![Feature_importance](figures/important_features.png)
	 The bar chart on the feature importance also indicates that the 'said' and 'via' features have substantial significance in influencing the model decision compared to other features. 

2. SHAP will is used for both global and local interpretability, and is well suited for complex tasks to provide a list of feature contributions.  It shows the class prediction score for each feature and the final class is selected based on the majority score. The class 1 indcates that it is real and 0 indicates that it is fake. 
![shap_local_explainability](figures/shap_local_explainability.png)

From the waterfall plot above, an idea on the features that the model relies on class prediction is evident. The 6th indexed test data shows that the features 'said','via', and 'washington' predicts the data as real. These are some of the important features whose prediction score is used by the model for prediction.

![shap_global_explainability](figures/shap_global_explainability.png)

 The above summary plot picturizes how the model works in a global scale. It shows that 'said' and 'via' are two important word features that the model heavily relies on deciding which class a data belongs to.

3. LIME on the other hand, is much better suited for localized interpretability and looking at individual predictions, especially in the context of text classification. In this image we can see the results of LIME analysis on a single news article. The words highlighted in orange are the terms that the model thinks contributes to deciding that the article is true, while the words in blue contribute to the article being classified as fake. The number next to the word indicates how much weight that word holds when deciding on the class prediction. We can see that based on the presence of certain words and their frequency, the model gives a prediction probability of 0.75 that this article is fake.

![LIME](figures/LIME.png)
 </p>

## Evaluation

To evaluate the classifier, we run classification report and compare the accuracy score. With the above mentioned 20 pairs, we could only get 15 pairs in the end, as some caused a negative result. We compare the 15 pairs and find the spaCy-based W2V-matrix plus SVM show the best accuracy score. Below are the bar chart of accuracy score.

![accuracy](figures/accuracy.png)  


## Conclusion
  
It was a great journey of analyzing a complex text data and retrieving insights. We as a team were able to grow our Machine Learning skills through implimenting various ML as well as other various data analysis techniques. The data we used was pretty large with around 44,000 rows and each data point in the 'text' column- the feature we used for the analysis was itself much large in size, adding complexity to the data. We experimented with a lot of text processing libraries and model explaining tools and libraries, which was very challenging at first, because we weren't exposed to those before. Finally, after immense research and analysis, we found that spaCy word2vec has the best result with SVM and RandomForest, and runs faster with an accuracy score of 96%. Also, using various model explainability techniques we were able to identify the working of classifier and the important features. Using spaCy, we were able to identify special words that were unique to fake news which included profanities. 


## Attribution  

<p>Abhina Premachandran Bindu - I did substantial work on researching on model explainability and implementing it for our project. Initially, I did a preliminary analysis of the data and build a classifier using nltk library. I also used gensim library to preprocess data and to build word2vec word embedding as our group decided to focus on different nlp libraries. I also build a python libary that can be imported to run a loop that measures the test and validation accuracies of each model to select the best one.  
</p>

<p>Esther Aruti - I contributed to exploring the dataset using the baseline NLTK, and then used spaCy to conclude that spaCy yielded better results. Additionally, I did the POS tagging and Entity recognition analysis along with their visualizations. I adjusted the entire dataset to remove the term 'reuters' which was impacting classification. I also contributed to doing the topic modeling with gensim by filtering out symbols and stop words as well as creating the visualization. Finally, I created the LIME analysis for model interpretability. I also largely contributed to creating the presentation powerpoint slides. 

Zhongming Wu - I was involved in Data Cleaning, Text Preprocessing, and Feature Extraction, dedicating most of my time to the latter. In addition to using NLTK, I explored word embedding algorithms, experimenting with several pre-trained models in spaCy, as well as gensim’s Word2Vec and Doc2Vec models. Although I couldn't get BERT to run successfully, the matrices from Word2Vec and Doc2Vec preserved more features and had lower dimensions compared to TF-IDF, providing a solid foundation for machine learning model training. I also used topic modeling with the gensim Doc2Vec model to interpret the dataset through topics and keywords.  

Mahbuba Jyoti - In our machine learning project focused on detecting fake news, I, Abhina Premachandran Bindu,Zhongming Wu,Esther Aruti played a crucial role in several key areas. Initially, I conducted a preliminary analysis of the data and built an initial classifier using the NLTK library. This early work helped set the foundation for our text processing pipeline. I also utilized the Gensim and SpaCy library to preprocess the data and developed word2vec word embeddings, as our group had decided to explore various NLP libraries to determine the most effective tools for our task.
A significant part of my contribution was researching model explainability and implementing it in our project. I developed a Python library that could be imported to run a loop, measuring the test and validation accuracies of each model. This systematic approach allowed us to efficiently compare the performance of different models and select the best one.
Additionally, I collaborated with my teammate on exploratory data analysis (EDA) to uncover patterns and insights that informed our approach. My efforts in text processing, specifically with NLTK, SpaCy Gensim, were pivotal in building a robust preprocessing pipeline that enhanced the overall performance of our models.
Through my focused contributions and collaborative efforts, we were able to determine that spaCy was superior for text processing and vector representation, and the SVA model was the most effective for detecting fake news. My work was instrumental in ensuring the success of our project, and I closely followed the directions set by our project lead, Zhongming Wu, to maintain alignment within the team.

<img src="https://github.com/DataScienceAndEngineering/machine-learning-dse-i210-final-project-fake-news/blob/main/reports/figures/fixed_github_commits.png" width="800"/>

## Bibliography

1. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7541057/
2. https://cits.ucsb.edu/fake-news/brief-history
3. https://spacy.io/api
4. https://radimrehurek.com/gensim/auto_examples/index.html#documentation
5. https://www.openlayer.com/blog/post/understanding-lime-in-5-steps#
6. https://www.datacamp.com/tutorial/introduction-to-shap-values-machine-learning-interpretability#
7. https://www.nltk.org/
8. https://textblob.readthedocs.io/en/dev/
9. https://fasttext.cc/
10. https://huggingface.co/blog/bert-101


## Appendix  

**Topic Modeling**

Apart from our main analysis, we also used Gensim LDAmodel, which is a convenient to calculate the similarity of document topics. Below is the practice of topic modeling applied to both fake and true articles split into two topics.  
From the left panel, it's easy to observe the similarity of topics. The further apart they are the more dissimilar they are. A larger circle indicates more distribution within that topic, and a smaller circle, less. From the right panel, we can observe the top ranking key word for each topic ordered by frequency. By adjusting the relevance metric slider at the top we can also see the words that were the least relevant to a particular topic (which could also provide some extra insight about the differences between the topics).

The visualization below is the topic modeling results of the original dataset, containing both fake and real news, and with stopwords and irrelevant symbols removed. The large distance between the two circles indicates a clear dissimilarity between the two topics. This makes sense, as the two main topics that explain the most variation within the dataset are articles that contain fake news (and the key words associated with it) and real news. Based on the most relevant terms (and our previous explorations of the data) we can infer that the topic labeled "1" is the fake news and the topic labeled "2" is the real news.

<img src="https://github.com/DataScienceAndEngineering/machine-learning-dse-i210-final-project-fake-news/blob/main/reports/faketrue_topic.gif" width="800"/>

Here we show topic modeling on only the fake news, split into four different topics. This would represent the main four topics found in fake articles. Looking at the top terms of the largest topic, topic "1" we can see words like "trump", "said", and "police".

<img src="https://github.com/DataScienceAndEngineering/machine-learning-dse-i210-final-project-fake-news/blob/main/reports/fake_topic.gif" width="800"/>


Now, we can also look at topic modeling done on true news, also split into four topics. These would represent the four main topics in true articles. We can see that in the largest topic, the word "said" is more relevant in comparison to "trump". This agrees with our previous findings that "said" tends to be more common in true news than in fake news.

<img src="https://github.com/DataScienceAndEngineering/machine-learning-dse-i210-final-project-fake-news/blob/main/reports/true_topic.gif" width="800"/>
