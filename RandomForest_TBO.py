# Import necessary libraries
import pandas as pd
import numpy as np


# Decision Tree helper functions

# Check if the provided data contains only one class
def check_purity(data):
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False

# Create a leaf node with the predicted value
def create_leaf(data, ml_task):
    label_column = data[:, -1]
    if ml_task == "regression":
        leaf = np.mean(label_column)
    else:
        unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
        index = counts_unique_classes.argmax()
        leaf = unique_classes[index]
    return leaf

# Get potential splits for each feature
def get_potential_splits(data):
    potential_splits = {}
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):  # excluding the last column which is the label
        values = data[:, column_index]
        unique_values = np.unique(values)
        potential_splits[column_index] = unique_values
    return potential_splits

# Calculate entropy for classification or mean squared error for regression
def calculate_entropy(data):
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
    return entropy

def calculate_mse(data):
    actual_values = data[:, -1]
    if len(actual_values) == 0:  # empty data
        mse = 0
    else:
        prediction = np.mean(actual_values)
        mse = np.mean((actual_values - prediction) ** 2)
    return mse

def calculate_overall_metric(data_below, data_above, metric_function):
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n
    overall_metric = (p_data_below * metric_function(data_below) + p_data_above * metric_function(data_above))
    return overall_metric
# Determine the best split based on the minimum entropy or mean squared error
def determine_best_split(data, potential_splits, ml_task):
    first_iteration = True
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)

            if ml_task == "regression":
                current_overall_metric = calculate_overall_metric(data_below, data_above, metric_function=calculate_mse)
            else:
                current_overall_metric = calculate_overall_metric(data_below, data_above, metric_function=calculate_entropy)

            if first_iteration or current_overall_metric <= best_overall_metric:
                first_iteration = False
                best_overall_metric = current_overall_metric
                best_split_column = column_index
                best_split_value = value

    return best_split_column, best_split_value

# 1.5 Split data
def split_data(data, split_column, split_value):
    split_column_values = data[:, split_column]
    type_of_feature = FEATURE_TYPES[split_column]
    if type_of_feature == "continuous":
        data_below = data[split_column_values <= split_value]
        data_above = data[split_column_values > split_value]
    else:
        data_below = data[split_column_values == split_value]
        data_above = data[split_column_values != split_value]
    return data_below, data_above

# Decision Tree Algorithm

# Determine the type of feature (categorical or continuous) for each column
def determine_type_of_feature(df):
    feature_types = []
    n_unique_values_treshold = 15
    for feature in df.columns:
        if feature != "class":
            unique_values = df[feature].unique()
            example_value = unique_values[0]
			# Check if the feature is categorical or continuous
            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")
    return feature_types

# Decision tree algorithm using recursive splitting
def decision_tree_algorithm(df, ml_task, counter=0, min_samples=2, max_depth=5):
    if counter == 0:
        global COLUMN_HEADERS, FEATURE_TYPES
        COLUMN_HEADERS = df.columns
        FEATURE_TYPES = determine_type_of_feature(df)
        data = df.values
    else:
        data = df

    # base cases
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        leaf = create_leaf(data, ml_task)
        return leaf

    # recursive part
    else:
        counter += 1

        # helper functions
        potential_splits = get_potential_splits(data)
        split_column, split_value = determine_best_split(data, potential_splits, ml_task)

        data_below, data_above = split_data(data, split_column, split_value)

        # check for empty data
        if len(data_below) == 0 or len(data_above) == 0:
            leaf = create_leaf(data, ml_task)
            return leaf

        # determine question
        feature_name = COLUMN_HEADERS[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        if type_of_feature == "continuous":
            question = "{} <= {}".format(feature_name, split_value)
        else:
            question = "{} = {}".format(feature_name, split_value)

        # instantiate sub-tree
        sub_tree = {question: []}

        # find answers (recursion)
        yes_answer = decision_tree_algorithm(data_below, ml_task, counter, min_samples, max_depth)
        no_answer = decision_tree_algorithm(data_above, ml_task, counter, min_samples, max_depth)

        # If the answers are the same, then there is no point in asking the question.
        # This could happen when the data is classified even though it is not pure
        # yet (min_samples or max_depth base case).
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)

        return sub_tree
    #Make predictions
# Predict a single example based on the decision tree
def predict_example(example, tree):
    # tree is just a root node
    if not isinstance(tree, dict):
        return tree

    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")

    # ask question
    if comparison_operator == "<=":
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # feature is categorical
    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer

    # recursive part
    else:
        residual_tree = answer
        return predict_example(example, residual_tree)

# Predict for all examples in a dataframe
def make_predictions(df, tree):
    if len(df) != 0:
        predictions = df.apply(predict_example, args=(tree,), axis=1)
    else:
        predictions = pd.Series()
    return predictions

# Calculate accuracy based on predictions
def calculate_accuracy(df, tree):
    predictions = make_predictions(df, tree)
    predictions_correct = predictions == df['class']
    accuracy = predictions_correct.mean()
    return accuracy



# Random Forest Helper Functions

# Subsample the data to create diverse trees
def subsample(data, ratio=1.0):
    sample_size = int(len(data) * ratio)
    sample = []
    while len(sample) < sample_size:
        index = np.random.randint(len(data))
        sample.append(data[index])
    return np.array(sample)

# Random forest algorithm that builds multiple decision trees
def random_forest_algorithm(train_df, ml_task, n_trees, subsample_ratio, max_depth):
    forest = []
    for _ in range(n_trees):
        subsampled_data = subsample(train_df.values, subsample_ratio)
        tree = decision_tree_algorithm(pd.DataFrame(subsampled_data, columns=train_df.columns), ml_task, max_depth=max_depth)
        forest.append(tree)
    return forest

# Make predictions using the random forest
def random_forest_predict(test_df, forest):
    predictions = []
    for tree in forest:
        tree_predictions = make_predictions(test_df, tree)
        predictions.append(tree_predictions)
    ensemble_predictions = np.array(predictions).transpose()
    final_predictions = []
    for row in ensemble_predictions:
        final_predictions.append(np.bincount(row.astype(int)).argmax())
    return np.array(final_predictions)

# Performance Scores Calculation
#  Calculate various performance scores based on predictions and actual labels

def calculate_performance_scores(predictions, actual_labels):
    accuracy = (predictions == actual_labels).mean()

    true_positive = np.sum((predictions == True) & (actual_labels == True))
    true_negative = np.sum((predictions == False) & (actual_labels == False))
    false_positive = np.sum((predictions == True) & (actual_labels == False))
    false_negative = np.sum((predictions == False) & (actual_labels == True))

    true_positive_rate = true_positive / (true_positive + false_negative)
    true_negative_rate = true_negative / (true_negative + false_positive)
    precision = true_positive / (true_positive + false_positive)
    f_score = 2 * (precision * true_positive_rate) / (precision + true_positive_rate)

    return accuracy, true_positive_rate, true_negative_rate, precision, f_score, true_positive, true_negative

# Data Preparation


# Load your train and test sets
train_df = pd.read_csv("trainSet.csv")  # Adjust the path to your train set
test_df = pd.read_csv("testSet.csv")    # Adjust the path to your test set    

# Assuming the last column is the label
# Adjust column names if needed
train_df.columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'class']
test_df.columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'class']

# Convert class labels to boolean (True/False)
train_df['class'] = train_df['class'] == 'good'
test_df['class'] = test_df['class'] == 'good'

# Train the Random Forest

ml_task = "classification"
n_trees = 50  # You can adjust the number of trees
subsample_ratio = 0.8
max_tree_depth = 6

forest = random_forest_algorithm(train_df, ml_task, n_trees, subsample_ratio, max_tree_depth)

# Make Predictions on Test Set using Random Forest

forest_test_predictions = random_forest_predict(test_df, forest)

# Calculate Performance Scores for Random Forest on Test Set

forest_test_accuracy, forest_test_tpr, forest_test_tnr, forest_test_precision, forest_test_fscore, forest_test_tp, forest_test_tn = calculate_performance_scores(forest_test_predictions, test_df['class'])

# Save the performance scores of the random forest to a text file

forest_report_file_path = "random_forest_performance_report_TBO.txt"

with open(forest_report_file_path, "w") as forest_report_file:
    forest_report_file.write("Random Forest Test results:\n")
    forest_report_file.write(f"Accuracy: {forest_test_accuracy:.3f}\n")
    forest_report_file.write(f"TPrate (Recall): {forest_test_tpr:.3f}\n")
    forest_report_file.write(f"TNrate: {forest_test_tnr:.3f}\n")
    forest_report_file.write(f"Precision: {forest_test_precision:.3f}\n")
    forest_report_file.write(f"F-Score: {forest_test_fscore:.3f}\n")
    forest_report_file.write(f"Total number of TP: {forest_test_tp}\n")
    forest_report_file.write(f"Total number of TN: {forest_test_tn}\n\n")
    forest_report_file.write("In this code, pandas library is used for:\n")
    forest_report_file.write("  - Loading and handling the training and testing datasets (pd.read_csv).\n")
    forest_report_file.write("  - Organizing and transforming the data into a format suitable for decision tree and random forest algorithms.\n")
    forest_report_file.write("  - Storing and manipulating the features and labels in a tabular format.\n\n")

    forest_report_file.write("And numpy library is used for:\n")
    forest_report_file.write("  - Numerical operations on arrays, such as mean calculations (np.mean).\n")
    forest_report_file.write("  - Handling and transforming data in a numerical format, which is crucial for machine learning algorithms.\n")
    forest_report_file.write("  - Efficiently working with potential splits and other numerical calculations in the decision tree and random forest algorithms.\n")


print(f"Random Forest Performance report saved to {forest_report_file_path}")
