# NLP-paper-implementation
A Novel Text Mining Approach Based on TF-IDF and Support Vector Machine for News Classification

---

# News Classification using TF-IDF and SVM

## Introduction
This repository implements a news classification method based on TF-IDF (Term Frequency-Inverse Document Frequency) and SVM (Support Vector Machine). The approach is inspired by the paper "A Novel Text Mining Approach Based on TF-IDF and Support Vector Machine for News Classification". The goal is to classify news articles into distinct categories to facilitate quick access to relevant information.

## Problem Statement and Objectives
The project addresses the challenge of automatically classifying news articles into different categories. With the vast amount of textual data available online, efficient tools are needed for organizing and retrieving information. News classification helps users identify articles of interest across categories such as business, entertainment, politics, sports, and technology. The proposed method uses TF-IDF for feature extraction and SVM for classification, evaluated primarily on the BBC dataset.

## State of the Art
Previous techniques in news classification include:
- **Financial News Classification**: Using machine learning algorithms to classify finance articles based on features like financial keywords and market trends.
  
- **Short Text Classification**: Specific approaches to handle the fragmented and concise nature of short texts, often found in social media.

- **Automatic Headline Classification**: Methods focusing on news headlines using techniques like TF-IDF and Bayesian classifiers.

- **Hybrid Text Classification Approaches**: Combination of multiple algorithms such as KNN and SVM to enhance classification accuracy.

- **Emotion Extraction from News Headlines**: Using NLP techniques to identify emotions conveyed in news headlines.

Each method has its own strengths and limitations. For instance, Bayesian algorithms, while effective in certain tasks, may lack precision in modeling texts. Hybrid approaches aim to combine the strengths of different techniques to improve results.

## Dataset
### BBC Dataset:
Consists of news articles from the BBC, categorized into five classes: business, entertainment, politics, sport, and tech.

### 20Newsgroup Dataset:
Originally intended for use, this dataset comprises articles from online discussion groups classified into 20 categories. However, it was inaccessible for our implementation.

## Methodology   

![image](https://github.com/najwanaamane/NLP-paper-implementation/assets/86806375/83fdc0b7-5b3b-4d5e-8545-f06b6034c9f7)

### Text Preprocessing
This step involves cleaning text data by removing unwanted elements such as punctuation, converting to lowercase, and filtering out stopwords (common words like "and", "the", etc.). This process simplifies texts and reduces data volume, enhancing model efficiency.
- *Note*: The paper utilized RapidMiner for data preprocessing, feature extraction, and model training, resulting in an efficient workflow for news article classification. In our implementation, we used Python libraries such as pandas, scikit-learn, matplotlib, and seaborn to perform necessary text preprocessing steps.

### Feature Extraction with TF-IDF
TF-IDF (Term Frequency-Inverse Document Frequency) calculates the importance of each word in a document relative to its frequency in that document and rarity across the entire corpus. This technique aids in identifying the most representative words in each text.

### Train-Test Split + SVM Classification
This step involves splitting the dataset into training and testing subsets. This separation verifies if the model can generalize its predictions on unseen data.
- SVM (Support Vector Machine) is a supervised classification algorithm that seeks to find the optimal hyperplane separating different classes to maximize the margin between closest data points. The RBF (radial basis function) kernel is used to handle nonlinear data by projecting it into a higher-dimensional space.

### Model Evaluation
Model performance is evaluated using precision, recall, and F1 score metrics. Precision measures the proportion of correct predictions among all predicted instances, recall measures the proportion of true positives detected among all actual positives, and F1 score is the harmonic mean of precision and recall.

### Results Interpretation
#### Overall Precision
The model achieves an overall precision of 97.84%, indicating high performance in classifying news articles into various categories. The overall F1 score is also high, suggesting a good balance between precision and recall.   
![image](https://github.com/najwanaamane/NLP-paper-implementation/assets/86806375/3abb6d60-1ea3-46f2-9bbc-a5935903c642)

#### Confusion Matrix
The confusion matrix shows that most categories are correctly classified. However, there are some misclassifications, such as certain "business" news articles being categorized incorrectly into other categories.   
![image](https://github.com/najwanaamane/NLP-paper-implementation/assets/86806375/3486047d-0100-47a4-801e-79f1f39ba567)


#### Precision and F1-Score Tables by Class
Precision and F1-score tables per class provide a detailed view of the model's performance for each news category, highlighting where the model excels and where improvements could be made.   

![image](https://github.com/najwanaamane/NLP-paper-implementation/assets/86806375/f41bf647-7480-45f4-aa5b-1ef55c49adaa)
![image](https://github.com/najwanaamane/NLP-paper-implementation/assets/86806375/8cf1e44d-3913-4fb2-8790-384f59908ee2)


### Interpretation of Results
The precision and F1-score tables per class indicate that categories like "sport" and "entertainment" perform exceptionally well, with precisions and F1-scores close to 1. Conversely, the "business" category shows lower performance, suggesting potential enhancements through additional features or SVM model parameter tuning.

## Critique of the Work
### Strengths
- **High Precision**: The paper demonstrates that the TF-IDF and SVM-based approach achieves high precision for both the BBC and 20Newsgroup datasets.
- **Simplicity and Effectiveness**: The method is relatively straightforward to implement while delivering competitive results, making it a practical solution for text classification tasks.

### Weaknesses
- **Dependency on RapidMiner**: The paper relies on RapidMiner for data preprocessing, which may limit reproducibility and adaptability for those without access to this tool.
- **Unbalanced Data**: The paper does not discuss the impact of class imbalance in the datasets used, which could significantly affect model performance.
- **Lack of Comprehensive Comparison**: While results are compared with other methods, the paper lacks detailed insights into exact configurations and hyperparameters used for these comparisons, making it challenging to fully evaluate the robustness of the proposed method.

---


For more details, you can refer to the original paper: [A Novel Text Mining Approach Based on TF-IDF and Support Vector Machine for News Classification](https://ieeexplore.ieee.org/document/7569223).

