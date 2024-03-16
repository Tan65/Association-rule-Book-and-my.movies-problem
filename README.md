# Association-rule-Book-and-my.movies-problem
Prepare rules for the all the data sets  1) Try different values of support and confidence. Observe the change in number of rules for different support,confidence values 2) Change the minimum length in apriori algorithm 3) Visulize the obtained rules using different plots 
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
data_book = pd.read_csv('Book1.csv')
data_movies = pd.read_csv('my_movies.csv')

# Convert datasets to binary format
data_book_binary = data_book.applymap(lambda x: 1 if x == 1 else 0)
data_movies_binary = data_movies.applymap(lambda x: 1 if x == 1 else 0)

# Function to perform association rule mining
def perform_association_rule_mining(data, min_support, min_confidence, min_length):
    frequent_itemsets = apriori(data, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules['antecedents'].apply(lambda x: len(x) >= min_length)]
    return rules

# Function to visualize association rules
def visualize_rules(rules, dataset_name):
    # Plotting support vs confidence
    plt.figure(figsize=(8, 6))
    plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Support vs Confidence - ' + dataset_name)
    plt.show()

    # Plotting support vs lift
    plt.figure(figsize=(8, 6))
    plt.scatter(rules['support'], rules['lift'], alpha=0.5)
    plt.xlabel('Support')
    plt.ylabel('Lift')
    plt.title('Support vs Lift - ' + dataset_name)
    plt.show()

    # Plotting distribution of support
    plt.figure(figsize=(8, 6))
    sns.histplot(rules['support'], bins=20, kde=True)
    plt.xlabel('Support')
    plt.ylabel('Frequency')
    plt.title('Distribution of Support - ' + dataset_name)
    plt.show()

# Set parameters
min_support = 0.1
min_confidence = 0.5
min_length = 2

# Perform association rule mining for book dataset
rules_book = perform_association_rule_mining(data_book_binary, min_support, min_confidence, min_length)
visualize_rules(rules_book, 'Book Dataset')

# Perform association rule mining for movies dataset
rules_movies = perform_association_rule_mining(data_movies_binary, min_support, min_confidence, min_length)
visualize_rules(rules_movies, 'Movies Dataset')
