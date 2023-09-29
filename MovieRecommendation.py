#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -qq convokit')


# In[2]:


get_ipython().system('pip install -qq scikit-surprise')


# In[3]:


import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import convokit


# In[4]:


# Load the Cornell Movie Dialogs Corpus
corpus = convokit.Corpus(filename=convokit.download("movie-corpus"))


# In[5]:


# Display basic statistics
print("Number of conversations:", len(corpus.conversations))
print("Number of users:", len(corpus.speakers))
print("Number of utterances:", len(corpus.utterances))


# In[6]:


# Display information about conversations
for convo_id in corpus.get_conversation_ids():
    convo = corpus.get_conversation(convo_id)
    print("Conversation ID:", convo_id)
    print("Metadata:", convo.meta)
    print("Number of utterances in conversation:", len(convo.get_utterance_ids()))
    print()


# In[7]:


# Create empty lists to store data
conversation_ids = []
movie_indices = []
movie_names = []
release_years = []
ratings = []
votes = []
genres = []
num_utterances = []

# Loop through conversations and extract data
for convo_id in corpus.get_conversation_ids():
    convo = corpus.get_conversation(convo_id)

    # Extract metadata from ConvoKitMeta object
    metadata = convo.meta

    # Append data to respective lists
    conversation_ids.append(convo_id)
    movie_indices.append(metadata['movie_idx'])
    movie_names.append(metadata['movie_name'])
    release_years.append(metadata['release_year'])
    ratings.append(metadata['rating'])
    votes.append(metadata['votes'])
    genres.append(metadata['genre'])
    num_utterances.append(len(convo.get_utterance_ids()))

# Create a DataFrame from the lists
data = {
    'Conversation ID': conversation_ids,
    'Movie Index': movie_indices,
    'Movie Name': movie_names,
    'Rating': ratings,
}

df = pd.DataFrame(data)


# In[8]:


df


# In[9]:


# Create a Reader object specifying the rating scale
reader = Reader(rating_scale=(1, 10))


# In[10]:


# Load the dataset into Surprise format
data = df[['Conversation ID','Movie Name','Rating']]
data = Dataset.load_from_df(data, reader)

# Split the data into training and testing sets (80% train, 20% test)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Create and train the SVD recommendation model
model = SVD()
model.fit(trainset)

# Evaluate the model on the testing data (calculate RMSE)
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)


# In[11]:


# Generate recommendations for a specific user (e.g., user_id='L236416')
user_id = 'L36547'
user_movies = df[df['Conversation ID'] == user_id]['Movie Name'].unique()

# Create a list of unrated movies for the user
all_movies = df['Movie Name'].unique()
unrated_movies = np.setdiff1d(all_movies, user_movies)

# Generate predictions for unrated movies
user_recommendations = []
for movie_id in unrated_movies:
    predicted_rating = model.predict(user_id, movie_id).est
    user_recommendations.append((movie_id, predicted_rating))

    # Sort recommendations by predicted rating (highest first)
user_recommendations.sort(key=lambda x: x[1], reverse=True)


# In[13]:


# Display the top n recommendations
top_n = 3
print(f'Top {top_n} recommendations for user {user_id}:')
for movie_id, predicted_rating in user_recommendations[:top_n]:
    print(f'Movie ID: {movie_id}, Predicted Rating: {predicted_rating:.2f}')


# In[ ]:





# In[ ]:




