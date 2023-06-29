# MADS_Milestone-2 : Predicting Text Difficulty
## <img width="500" height="400" src="https://github.com/cheerspankaj/MADS_Milestone-2/assets/82276130/26098ba1-dd47-4755-8882-2a01cdc7db73" > <img width="500" height="400" src="https://www.unicef.org/bulgaria/sites/unicef.org.bulgaria/files/styles/media_banner/public/%23LetMeLearn.png?itok=SFNyMo94">

Image source Link: https://www.unicef.org/bulgaria/en/press-releases/unicef-only-third-10-year-olds-globally-are-estimated-be-able-read-and-understand

Image source Link: https://www.unicef.org/bulgaria/en/let-me-learn

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

<img width="748" alt="image" src="https://github.com/cheerspankaj/MADS_Milestone-2/assets/82276130/a460fadc-5165-4a6e-8d40-811d1a2cb6e3">

The problem at hand involves binary text classification. We assessed several supervised learning classification models to predict the class labels. We started with a Dummy classifier as a baseline model and compared it with other models. For linear models, we applied Logistic Regression and used the TF-IDF feature for comparison. We also utilized Naive Bayes (BernoulliNB) classifiers, which are efficient for text data classification.

Next, we implemented ensemble decision tree classifiers, namely RandomForest and XGBoost, which combine multiple models to create powerful classifiers. These models address overfitting and leverage hierarchical partitions among categories.

We also explored a Neural Network model, specifically the Multilayer Perceptron (MLP), which achieved competitive scores with the available training data. This sparked our interest in using pre-trained models like BERT for further assessment, which is discussed in the subsequent sections.

### Hyper-Parameter Tuning

To optimize the model's performance, we conducted hyperparameter tuning using GridSearchCV. This involved trying different combinations of values to identify the model with the lowest error metric. By employing GridSearchCV for the RandomForest classifier and exploring various regularization parameters, we achieved improved accuracy scores of 0.71. 

The selected parameters for hyperparameter tuning included 'criterion' (gini or entropy), 'bootstrap' (True), 'max_features' (auto or sqrt), and 'n_estimators' (100 or 200). Additionally, based on sensitivity analysis, we allowed the model to determine the 'max_depth' parameter.

### Model Evaluation and Performance

Based on the initial analysis, Random forest classifier selected for evaluation pipelines. Below figures show the model evaluation analysis.

<img width="847" alt="image" src="https://github.com/cheerspankaj/MADS_Milestone-2/assets/82276130/bc3fc2cc-ecf3-45ef-a6ad-eade3d412375">

Figure: Classification Report and Confusion Matrix for Random Forest Classifier, source: Project Code

<img width="816" alt="image" src="https://github.com/cheerspankaj/MADS_Milestone-2/assets/82276130/6b227e9c-c0d8-4dd7-92dc-a8efe3984bc2">

Figure: Precision Recall and ROC-AUC curve for Random Forest Classifier, Source: Project Code

### Failure Analysis

For Logistic Regression, Random Forest, and BERT models, we employed Local Interpretable Model-Agnostic Explanations (LIME) to gain insights into local model interpretability. LIME operates by modifying individual data samples and observing how these modifications impact the output. Our goal was to understand why specific predictions were made and identify the influential variables. The output provided by LIME highlights the contribution of the top-6 features towards the model predictions. Please refer project report section "Failure Analysis" for more details.

### Sensitivity Analysis

To assess the model's sensitivity and generalizability, we conducted sensitivity analysis using the best extracted parameters. Specifically, we focused on max_depth and n_estimators as the parameters of interest. By plotting these parameters against the accuracy scores, we found that the model performed optimally with a max depth of 9. Choosing a higher max depth led to a drop in the score, indicating the risk of overfitting or underfitting. Additionally, when analyzing the hyperparameter n_estimators, we noticed a slight increase in the score at a value of 300, without any change in the training score.

<img width="784" alt="image" src="https://github.com/cheerspankaj/MADS_Milestone-2/assets/82276130/0f7c30a1-0b0d-4815-a16d-01990fcbbb10">

Figure: Sensitivity Analysis with max_depth and n_estimators, source: Project Code

### Learning Curve

The learning curve analysis involved plotting the amount of training data on the x-axis and evaluating the metric of interest. The objective was to determine if increasing the amount of data would improve the overall evaluation metric and understand the impact of training data variability on classifier predictions. Despite considering split datasets, it was observed that as the dataset size increased, the score also increased. This suggests that feeding more data to the classifier is likely to enhance its performance.

<img width="458" alt="image" src="https://github.com/cheerspankaj/MADS_Milestone-2/assets/82276130/e4f37ebc-81c8-4255-a458-2859c4133bf5">

Figure: Learning Curve for Random Forest Classifier, source: Project Code

