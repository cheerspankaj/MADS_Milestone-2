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
