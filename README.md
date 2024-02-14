# MADS_Milestone-2 : Predicting Text Difficulty
## <img width="400" height="300" src="https://github.com/cheerspankaj/MADS_Milestone-2/assets/82276130/26098ba1-dd47-4755-8882-2a01cdc7db73" > <img width="400" height="300" src="https://www.unicef.org/bulgaria/sites/unicef.org.bulgaria/files/styles/media_banner/public/%23LetMeLearn.png?itok=SFNyMo94">

Image source Link: https://www.unicef.org/bulgaria/en/press-releases/unicef-only-third-10-year-olds-globally-are-estimated-be-able-read-and-understand

Image source Link: https://www.unicef.org/bulgaria/sites/unicef.org.bulgaria/files/styles/media_banner/public/%23LetMeLearn.png.webp?itok=35iFgTEq

## Objectives
The objective of this project is to explore the use of text simplification applications. Specifically, we focus on determining if a given textbook is appropriate for a student's reading level prior to assigning it as homework. In this paper, we present an analysis of various supervised and unsupervised methods for predicting text difficulty. For more information on the problem, please refer to the Kaggle competition.
Data source: [Link](https://www.kaggle.com/t/5a1872e494574cc7bbf433fa8f4687d9).

## Datasets
The primary dataset used for training the model is "WikiLarge_Train.csv", containing 416,768 english (mostly English but does contain
other languages as well) sentences with a label for each sentence.

0: the sentence does NOT need to be simplified.

1: the sentence DOES need to be simplified.

The test data contains “WikiLarge_Test.csv” 119,092 sentences that are unlabeled.
data source:[Link](https://www.kaggle.com/competitions/umich-siads-696-f22-predicting-text-difficulty/data)

3rd party datasets: 
provided by the Kaggle competition website:
- dale_chall.txt : “This is the Dale-Chall list of ~3000 elementary English words that are typically familiar to 80% of American 4th grade students (in the 90s)”
data source: [Link](https://www.kaggle.com/competitions/umich-siads-696-f22-predicting-text-difficulty)
- Concreteness_ratings_Brysbaert_et_al_BRM.txt : “Concreteness ratings for about 40k English words."
  Data Source: [Link](https://www.kaggle.com/competitions/umich-siads-696-f22-predicting-text-difficulty)
- AoA_51715_words.csv :: “List of approximate age (In years) when a word was learned, for 50k English words.”
  Data Source: [Link](https://www.kaggle.com/competitions/umich-siads-696-f22-predicting-text-difficulty)

## Data Preparation and Feature Engineering

Our data preparation and feature engineering process began with a manual review of the datasets. This involved examining the dataset's columns, assessing balance, identifying missing values, and determining the best features for modeling purposes.

During preprocessing, we applied commonly used techniques in NLP pipelines such as lowercasing, removal of special characters, stopwords, sentence tokenization, and words lemmatization. This step aimed to clean the data, perform analysis, and generate new features from the existing dataset while evaluating their impact on model performance.

Since the problem involved text classification, we explored different approaches to convert text into feature vectors, including CountVectorizer, TF-IDF (Term Frequency-Inverse Document Frequency), and Doc2Vec.

We initially extracted 20 features from the provided datasets (listed in Appendix - 2.a of the project report) and narrowed it down to thirteen features (listed in Appendix - 2.b of the project report) through feature ablation. These features were used to build supervised and unsupervised learning models for the binary classification prediction problem.

Finally, we transformed the entire dataset of features into a vectorized form readable by machine learning algorithms for text classification purposes.

## Machine Learning Models

Following Machine Learning models developed for model selection and uploaded to this repository.

- Dummy Classifier
- Decision Tree
- Naive Bayes
- Gradient Boositng
- Logistic Regression 
- Random Forest
- Neural Network MLP

<img width="750" alt="image" src="https://github.com/cheerspankaj/MADS_Milestone-2/assets/82276130/06e5b86b-7545-4ce6-bb72-e71fd1885d32">

*Table: Summarizing the accuracy score for all models assessed*

The problem at hand involves binary text classification. We assessed several supervised learning classification models to predict the class labels. We started with a Dummy classifier as a baseline model and compared it with other models. For linear models, we applied Logistic Regression and used the TF-IDF feature for comparison. We also utilized Naive Bayes (BernoulliNB) classifiers, which are efficient for text data classification.

Next, we implemented ensemble decision tree classifiers, namely RandomForest and XGBoost, which combine multiple models to create powerful classifiers. These models address overfitting and leverage hierarchical partitions among categories.

We also explored a Neural Network model, specifically the Multilayer Perceptron (MLP), which achieved competitive scores with the available training data. This sparked our interest in using pre-trained models like BERT for further assessment, which is discussed in the subsequent sections.

### Hyper-Parameter Tuning

To optimize the model's performance, we conducted hyperparameter tuning using GridSearchCV. This involved trying different combinations of values to identify the model with the lowest error metric. By employing GridSearchCV for the RandomForest classifier and exploring various regularization parameters, we achieved improved accuracy scores of 0.71. 

The selected parameters for hyperparameter tuning included 'criterion' (gini or entropy), 'bootstrap' (True), 'max_features' (auto or sqrt), and 'n_estimators' (100 or 200). Additionally, based on sensitivity analysis, we allowed the model to determine the 'max_depth' parameter.

### Model Evaluation and Performance

Based on the initial analysis, Random forest classifier selected for evaluation pipelines. Below figures show the model evaluation analysis.

<img width="847" alt="image" src="https://github.com/cheerspankaj/MADS_Milestone-2/assets/82276130/bc3fc2cc-ecf3-45ef-a6ad-eade3d412375">

*Figure: Classification Report and Confusion Matrix for Random Forest Classifier, source: Project Code*

<img width="816" alt="image" src="https://github.com/cheerspankaj/MADS_Milestone-2/assets/82276130/6b227e9c-c0d8-4dd7-92dc-a8efe3984bc2">

*Figure: Precision Recall and ROC-AUC curve for Random Forest Classifier, Source: Project Code*

### Failure Analysis

For Logistic Regression, Random Forest, and BERT models, we employed Local Interpretable Model-Agnostic Explanations (LIME) to gain insights into local model interpretability. LIME operates by modifying individual data samples and observing how these modifications impact the output. Our goal was to understand why specific predictions were made and identify the influential variables. The output provided by LIME highlights the contribution of the top-6 features towards the model predictions. Please refer project report section "Failure Analysis" for more details.

### Sensitivity Analysis

To assess the model's sensitivity and generalizability, we conducted sensitivity analysis using the best extracted parameters. Specifically, we focused on max_depth and n_estimators as the parameters of interest. By plotting these parameters against the accuracy scores, we found that the model performed optimally with a max depth of 9. Choosing a higher max depth led to a drop in the score, indicating the risk of overfitting or underfitting. Additionally, when analyzing the hyperparameter n_estimators, we noticed a slight increase in the score at a value of 300, without any change in the training score.

<img width="784" alt="image" src="https://github.com/cheerspankaj/MADS_Milestone-2/assets/82276130/0f7c30a1-0b0d-4815-a16d-01990fcbbb10">

*Figure: Sensitivity Analysis with max_depth and n_estimators, source: Project Code*

### Learning Curve

The learning curve analysis involved plotting the amount of training data on the x-axis and evaluating the metric of interest. The objective was to determine if increasing the amount of data would improve the overall evaluation metric and understand the impact of training data variability on classifier predictions. Despite considering split datasets, it was observed that as the dataset size increased, the score also increased. This suggests that feeding more data to the classifier is likely to enhance its performance.

<img width="416" alt="image" src="https://github.com/cheerspankaj/MADS_Milestone-2/assets/82276130/38246e02-ac1e-457e-93ec-dee2a26b2fe9">

*Figure: Learning Curve for Random Forest Classifier, source: Project Code*

## BERT Pre-Trained Model

To enhance the overall performance of our model and explore advanced models in NLP, we incorporated BERT pretrained models from Hugging Face. BERT, an acronym for Bidirectional Encoder Representations from Transformers, is a recent model in the field of NLP designed to assist with various language tasks, including text classification, question answering, abstract text summarization, translation, and more.

In our analysis of the BERT model, we explored various parameters and configurations. The BERT "large" version consistently outperformed the "base" version by around 1-2 percentage points in terms of accuracy. However, we encountered memory issues when using a batch size of 128 and sentence length of 128 on a single GPU. We found that a batch size of 64 and sentence length of 64 yielded slightly better results than a batch size and sentence length of 32. Comparing the regular BERT model with the multilingual version, we observed minimal differences in overall accuracy. Surprisingly, the parameter that had a significant impact on the model's performance was the dropout rate. Dropout rates of 50% or higher resulted in a drastic decrease in accuracy, resembling random guessing. The best overall score we achieved was with a 1% dropout rate and a batch size and sentence length of 64 using the BERT "large" version. This yielded a score of 80.949% on Kaggle, close to the 81% mark. To further improve the score using BERT, future efforts should consider utilizing BERT "large" with a batch size and sentence length of 128 or higher, as well as leveraging multiple GPUs. Our analysis of a randomly selected subset of 100,000 sentences confirmed that a dropout rate of 50% significantly degraded the validation accuracy from approximately 70% to 58%.

<img width="825" alt="image" src="https://github.com/cheerspankaj/MADS_Milestone-2/assets/82276130/c0f76fdf-a8e7-43a3-9b3a-9e932f6e7ccc">

*Figures: BERT validation performance on 100,000 random sentences, source: Project Code*

## Unsupervised Learning

### Principle Component Analysis (PCA)

PCA (Principal Component Analysis) was used to visualize high-dimensional data in two dimensions and identify patterns in the provided training dataset. The hypothesis was that there might be mixed language sentences, such as French, in the dataset, which would appear differently clustered in a two-dimensional space. Regular PCA and Kernel PCA with a radial basis function were applied, but the results were similar. The overall class separation in PCA was not ideal, but clear patterns were observed in the right part of the graph, indicating sentences in different languages like French, German, Italian, and Spanish. It was unclear if mixed languages were present in the other part of the graph where classes overlapped. The purpose of PCA was to identify patterns and explore the potential for improving supervised approaches in predicting text difficulty. Further analysis is needed to separate multi-language sentences and build more accurate classifiers. Excluding English stop words compressed the two components even further, suggesting that removing stop words resulted in fewer features and potentially clearer patterns. Both visualizations showed a "V" pattern, which could be attributed to TF-IDF or other patterns in the Kaggle dataset. The "V" pattern was more pronounced in the visualization without stop words due to reduced noise.

<img width="736" alt="image" src="https://github.com/cheerspankaj/MADS_Milestone-2/assets/82276130/59feb90b-e69c-4c26-a740-945ff70ae984">

*Figure1: PCA, N=2, Includes stop words    Figure2: PCA, N=2, Excludes stop words*
*Blue Color = difficult/ label as 1, Orange Color = Simple / label as 0  source: Project Code*

### Truncated Singular Value Decomposition (Truncated SVD)

We also performed truncated SVD on the TF-IDF vectorizer training data set to look for visual patterns. Unfortunately, this resulted in similar patterns found above within PCA. The graphical results look very similar however they appear to be mostly rotated 180 degrees compared to the PCA plots above.

<img width="647" alt="image" src="https://github.com/cheerspankaj/MADS_Milestone-2/assets/82276130/163f8e8b-02ac-460b-b727-e5df3b1d2b1f">

*Figure1: PCA, N=2, Includes stop words    Figure2: PCA, N=2, Excludes stop words*
*blude color = difficult / label as 1, orange color = simple / label as 0 source: Project Code*

### Latent Dirichlet Allocation (LDA)

LDA (Latent Dirichlet Allocation) was used for topic modeling, which aims to discover abstract topics from a collection of documents. The focus was on extracting high-quality, clear, and meaningful topics. The process involved data preprocessing, training the LDA model using Gensim, computing perplexity and coherence values, determining the optimal number of topics, visualizing topic allocation, and analyzing topics and terms using pyLDAvis. The dominant topic for each document in the training dataset was derived, and a comparison was made between logistic regression models with and without the topic feature, considering model score, precision, and recall.

<img width="803" alt="image" src="https://github.com/cheerspankaj/MADS_Milestone-2/assets/82276130/4216079f-ba64-4041-9ee5-5ae20a0ac02d">

*Figure: Top 40 Topics and associated Terms Visualization using pyLDAvis source:Project code*

#### Insights
Analysis result suggests that adding sentence topic as a feature from LDA topic modeling did not help to improve Logistic regression model score, precision and recall value.

<img width="748" alt="image" src="https://github.com/cheerspankaj/MADS_Milestone-2/assets/82276130/cb50cf1a-7a03-4186-b201-1d4011ee7518">

*Figure1: Precision-Recall without LDA topic   Figure2: Precision-Recall with LDA Topics source: Project Code*

### Clustering

#### K-Means

We aimed to identify clusters within a cleaned text dataset. The dataset was transformed into a sparse format using TFIDF vectorization, resulting in a random sample of 50,000 training instances with approximately 2,000 features. To determine the appropriate number of clusters (K) for K-means clustering, we employed "The Elbow Method" technique.

<img width="688" alt="image" src="https://github.com/cheerspankaj/MADS_Milestone-2/assets/82276130/f2505bce-ea9d-458e-af4c-abd4b2ee2425">

*source: Project Code*

K-means visualization does not show any clear clusters between simple and complex sentences. Our understanding is that the features derived from the documents are not very similar and hence not able to form distinct clusters, we would like to further investigate to get more insights.

#### T-SNE & MDS

In high-dimensional data analysis, it is crucial to explore dimensionality-reduced datasets obtained from techniques like t-SNE, UMAP, or MDS. We applied t-SNE, UMAP, and MDS to a random sample of 10,000 instances from the training dataset, which had been transformed into a vectorized form with 380 features (dimensions).

<img width="790" alt="image" src="https://github.com/cheerspankaj/MADS_Milestone-2/assets/82276130/3d92729d-24d6-41fe-aa9f-1728ca6022a5">

*Figure1: T-SNE visualization   Figure2: MDS Visualization*

The T-SNE and MDS visualizations did not reveal any distinct clusters or patterns in the lower-dimensional feature space. However, further investigation is needed to understand the empty region between the outer ring and the center cluster in the T-SNE visualization. Both T-SNE and MDS showed that features from both simple and not simple sentences are scattered across all regions, indicating a lack of clear differentiation.

Project Git Repository link - https://github.com/cheerspankaj/MADS_Milestone-2
