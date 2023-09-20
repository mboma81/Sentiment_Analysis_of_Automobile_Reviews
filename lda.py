import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV data into a DataFrame
data = pd.read_csv('your_bag_of_words_dataset.csv')

# Perform Latent Dirichlet Allocation (LDA)
num_topics = 5  # Define the number of topics you want to extract
lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
X_lda = lda_model.fit_transform(data)

# Assign the most likely topic to each document
data['topic'] = X_lda.argmax(axis=1)

# Plotting the distribution of topics
plt.figure(figsize=(10, 6))
sns.countplot(x='topic', data=data)
plt.xlabel('Topic')
plt.ylabel('Count')
plt.title('Distribution of Topics')
plt.xticks(range(num_topics), [f"Topic {i+1}" for i in range(num_topics)])
plt.tight_layout()
plt.show()

# Plotting the top words for each topic
plt.figure(figsize=(12, 8))
for topic_idx, topic in enumerate(lda_model.components_):
    top_words = [data.columns[i] for i in topic.argsort()[:-10 - 1:-1]]  # Change 10 to the desired number of top words
    plt.barh(range(10), topic.argsort()[:-10 - 1:-1], align='center', color='skyblue')
    plt.yticks(range(10), top_words)
    plt.xlabel('Word Weight')
    plt.title(f'Top Words in Topic {topic_idx + 1}')
    plt.gca().invert_yaxis()  # Invert y-axis for better visualization
    plt.tight_layout()
    plt.show()
