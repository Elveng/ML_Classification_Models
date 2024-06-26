# ML_Classification_Models

This repository contains implementations of decision tree and random forest classifiers, developed as part of a Machine Learning lecture. The project includes scripts for training the models, making predictions, and evaluating their performance.

## Table of Contents
- [Introduction](#introduction)
- [Project Purpose](#project-purpose)
- [Part 1: Decision Tree](#part-1-decision-tree)
- [Part 2: Random Forest](#part-2-random-forest)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Introduction
This repository contains two main parts: a decision tree classifier and a random forest classifier. The goal is to provide hands-on experience with different classification algorithms and demonstrate their applications using Python.

## Project Purpose
The primary purpose of this project is to apply machine learning classification techniques to solve practical problems. It includes building and training decision tree and random forest classifiers to predict the class labels of a dataset. The project demonstrates the use of various libraries and frameworks for data manipulation, model training, and evaluation.

## Part 1: Decision Tree
The first part of the project involves building a decision tree classifier to predict the class label of data points. Decision trees are a popular machine learning method used for classification and regression tasks. They work by splitting the data into subsets based on the value of input features. This process is repeated recursively, forming a tree structure with decision nodes and leaf nodes.

### Implementation Details:
- **Data Loading and Preparation:** The training and test datasets are loaded using `pandas`. The data is organized into features and labels.
- **Model Training:** The decision tree algorithm is implemented to recursively split the data into subsets based on the most informative features. The algorithm stops when it reaches a specified maximum depth, the data is pure (all instances belong to one class), or there are too few samples to split further.
- **Prediction:** Once trained, the model is used to predict the class labels for the test data.
- **Performance Evaluation:** The model's performance is evaluated using metrics such as accuracy, precision, recall (TPrate), true negative rate (TNrate), and F-score. These metrics help assess the model's ability to correctly classify instances in the test dataset.

### Performance Metrics:
- **Train Results:**
  - Accuracy: 0.792
  - TPrate (Recall): 0.673
  - TNrate: 0.879
  - Precision: 0.805
  - F-Score: 0.733
  - Total number of TP: 140
  - Total number of TN: 248
- **Test Results:**
  - Accuracy: 0.735
  - TPrate (Recall): 0.586
  - TNrate: 0.881
  - Precision: 0.829
  - F-Score: 0.686
  - Total number of TP: 58
  - Total number of TN: 89

## Part 2: Random Forest
The second part of the project involves implementing a random forest classifier to predict the class label of data points. Random forests are an ensemble learning method that combines multiple decision trees to improve the model's overall performance. Each tree in the forest is trained on a random subset of the data, and the final prediction is made by aggregating the predictions of all trees (e.g., by majority voting for classification).

### Implementation Details:
- **Data Loading and Preparation:** Similar to the decision tree, the training and test datasets are loaded using `pandas`. The data is organized into features and labels.
- **Model Training:** The random forest algorithm is implemented to create multiple decision trees. Each tree is trained on a random subset of the data and features, ensuring diversity among the trees. The maximum depth of the trees is controlled to prevent overfitting.
- **Prediction:** The model aggregates the predictions of all decision trees to determine the final class label for each instance in the test data.
- **Performance Evaluation:** The performance of the random forest model is evaluated using the same metrics as the decision tree. This helps compare the effectiveness of the ensemble method against a single decision tree.

### Performance Metrics:
- **Test Results:**
  - Accuracy: 0.750
  - TPrate (Recall): 0.566
  - TNrate: 0.931
  - Precision: 0.889
  - F-Score: 0.691
  - Total number of TP: 56
  - Total number of TN: 94

## Contributing
We welcome contributions to improve this project. Please follow these steps to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Create a new Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
This project was developed as part of a Machine Learning lecture. Special thanks to the course instructors and peers for their support and feedback.

## Contact
For any inquiries, please contact:
- Name: Tahsin Berk ÖZTEKİN
- Email: tahsinberkoztekin@gmail.com
