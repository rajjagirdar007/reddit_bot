import praw
import pandas as pd
import numpy as np
import openai
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import time
import os
#from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
# Load environment variables


# Set up logging
logging.basicConfig(filename='reddit_bot.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Reddit API credentials
reddit = praw.Reddit(
    client_id=os.environ.get('REDDIT_CLIENT_ID'),
    client_secret=os.environ.get('REDDIT_CLIENT_SECRET'),
    user_agent=os.environ.get('REDDIT_USER_AGENT'),
    username=os.environ.get('REDDIT_USERNAME'),
    password=os.environ.get('REDDIT_PASSWORD')
)

print(os.environ.get('REDDIT_CLIENT_ID'))
print(os.environ.get('REDDIT_CLIENT_SECRET'))
print(os.environ.get('REDDIT_USER_AGENT'))
print(os.environ.get('REDDIT_USERNAME'))
print(os.environ.get('REDDIT_PASSWORD'))


# OpenAI API key
openai.api_key = os.environ.get('OPENAI_API_KEY')

from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

def train_subreddit_model(data, subreddit_name):
    # Prepare the data
    data['full_text'] = data['title'] + ' [SEP] ' + data['selftext']
    data['full_text'] = data['full_text'].apply(lambda x: x.replace('\n', ' '))
    
    # Save the data to a file
    with open(f"{subreddit_name}_training_data.txt", "w", encoding="utf-8") as f:
        for text in data['full_text']:
            f.write(text + "\n")
    
    # Load pre-trained model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare the dataset
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=f"{subreddit_name}_training_data.txt",
        block_size=128
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=f"./results_{subreddit_name}",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    model.save_pretrained(f"{subreddit_name}_model")
    tokenizer.save_pretrained(f"{subreddit_name}_tokenizer")
    
    return model, tokenizer

# Data Collection
def fetch_subreddit_data(subreddit_name, limit=1000):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    
    try:
        for post in subreddit.new(limit=limit):
            posts.append({
                'title': post.title,
                'selftext': post.selftext,
                'score': post.score,
                'num_comments': post.num_comments,
                'created': datetime.fromtimestamp(post.created_utc),
                'author': str(post.author),
                'upvote_ratio': post.upvote_ratio,
                'is_original_content': post.is_original_content,
                'over_18': post.over_18,
                'spoiler': post.spoiler,
                'stickied': post.stickied,
                'flair': post.link_flair_text
            })
        
        df = pd.DataFrame(posts)
        df['full_text'] = df['title'] + ' ' + df['selftext']
        df['day_of_week'] = df['created'].dt.dayofweek
        df['hour_of_day'] = df['created'].dt.hour
        df['text_length'] = df['full_text'].str.len()
        
        return df
    except Exception as e:
        logging.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

def analyze_data(data, subreddit_name):
    plt.figure(figsize=(12, 6))
    sns.histplot(data['score'], bins=50, kde=True)
    plt.title('Distribution of Post Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.savefig(f'{subreddit_name}_score_distribution.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='text_length', y='score', data=data)
    plt.title('Relationship between Text Length and Score')
    plt.xlabel('Text Length')
    plt.ylabel('Score')
    plt.savefig(f'{subreddit_name}_length_vs_score.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='day_of_week', y='score', data=data)
    plt.title('Score Distribution by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Score')
    plt.savefig(f'{subreddit_name}_score_by_day.png')
    plt.close()

    correlation_matrix = data[['score', 'num_comments', 'upvote_ratio', 'text_length', 'day_of_week', 'hour_of_day']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig(f'{subreddit_name}_correlation_matrix.png')
    plt.close()

    logging.info(f"Data analysis for r/{subreddit_name} completed. Visualizations saved.")


# Machine Learning
def train_model(data):
    X = data[['title', 'selftext', 'day_of_week', 'hour_of_day', 'text_length', 'is_original_content', 'over_18', 'spoiler', 'stickied']]
    y = data['score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X_train['title'] + " " + X_train['selftext'], y_train)

    y_pred = pipeline.predict(X_test['title'] + " " + X_test['selftext'])
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logging.info(f"Model Performance - MSE: {mse}, R2 Score: {r2}")

    joblib.dump(pipeline, 'reddit_model.pkl')
    logging.info("Model trained and saved.")

    return pipeline

# Content Generation
# Content Generation
def generate_post(model, prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates engaging Reddit posts."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            n=3,
            stop=None,
            temperature=0.7,
        )

        generated_posts = [choice['message']['content'].strip() for choice in response['choices']]
        scores = [model.predict([post])[0] for post in generated_posts]
        best_post = generated_posts[scores.index(max(scores))]

        return best_post
    except Exception as e:
        logging.error(f"Error generating post: {str(e)}")
        return ""

def get_subreddit_flairs(subreddit_name):
    subreddit = reddit.subreddit(subreddit_name)
    flairs = []
    try:
        for flair in subreddit.flair.link_templates:
            flairs.append(flair['id'])
        return flairs
    except Exception as e:
        logging.error(f"Error fetching flairs for r/{subreddit_name}: {str(e)}")
        return flairs



def create_prompt(data, subreddit_name):
    # Filter top performing posts
    top_posts = data[data['score'] > data['score'].quantile(0.9)]
    
    # Combine title and selftext
    text = ' '.join(top_posts['title'] + ' ' + top_posts['selftext'])
    
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    filtered_words = [word for word in word_tokens if word.isalnum() and word not in stop_words]
    
    # Get most common words
    word_freq = Counter(filtered_words)
    common_words = [word for word, _ in word_freq.most_common(10)]
    
    # Analyze best posting time
    best_day = data.groupby('day_of_week')['score'].mean().idxmax()
    best_hour = data.groupby('hour_of_day')['score'].mean().idxmax()
    
    # Analyze optimal length
    median_length = data['text_length'].median()
    
    # Generate prompt
    prompt = f"""Generate a Reddit post for r/{subreddit_name} that is likely to go viral. 
    Use these insights from data analysis:
    1. Include some of these popular topics/words: {', '.join(common_words)}
    2. The best day to post is {best_day} and the best hour is {best_hour}
    3. Aim for a post length of around {median_length} characters
    4. Posts with higher upvote ratios tend to perform better
    5. Make it funny and try to be as casual as possible
    
    Create a post that:
    - Has an engaging and descriptive title
    - Provides valuable information or sparks discussion
    - Is relevant to the r/{subreddit_name} community
    - Follows the subreddit's rules and guidelines
    - Is original and not a repost of common topics
    
    Format the response as:
    Title: [Generated Title]
    Content: [Generated Content]
    """
    
    return prompt


# Posting to Reddit
import random

# Posting to Reddit with Flair
def post_to_reddit(subreddit_name, title, content):
    try:
        subreddit = reddit.subreddit(subreddit_name)
        flairs = get_subreddit_flairs(subreddit_name)
        
        if not flairs:
            logging.error(f"No flairs available for r/{subreddit_name}. Skipping post.")
            return None
        
        flair_id = random.choice(flairs)
        
        post = subreddit.submit(title, selftext=content, flair_id=flair_id)
        logging.info(f"Posted to Reddit: {post.url}")
        return post
    except Exception as e:
        logging.error(f"Error posting to Reddit: {str(e)}")
        return None


# Monitoring Posts
def monitor_posts(subreddit_name, post_limit=10):
    subreddit = reddit.subreddit(subreddit_name)
    for post in subreddit.new(limit=post_limit):
        logging.info(f"Title: {post.title}, Score: {post.score}, Comments: {post.num_comments}")

# Main workflow
def main():
    # List of subreddits to analyze and post to
    subreddits = ['developersIndia']
    
    for subreddit_name in subreddits:
        logging.info(f"Processing subreddit: r/{subreddit_name}")
        
        # Data Collection
        data = fetch_subreddit_data(subreddit_name)
        if data.empty:
            logging.error(f"Failed to fetch data for r/{subreddit_name}. Skipping.")
            continue

        data.to_csv(f'{subreddit_name}_posts.csv', index=False)
        logging.info(f"Collected {len(data)} posts from r/{subreddit_name}")

        # Data Analysis
        analyze_data(data, subreddit_name)

        # Machine Learning
        model = train_model(data)

        # Content Generation and Posting
        prompt = create_prompt(data, subreddit_name)
        generated_content = generate_post(model, prompt)
        
        if generated_content:
            # Split the generated content into title and body
            title, content = generated_content.split('\n', 1)
            title = title.replace('Title: ', '').strip()
            content = content.replace('Content: ', '').strip()
            
            post = post_to_reddit(subreddit_name, title, content)
            
            if post:
                # Monitor the posted content
                time.sleep(3600)  # Wait for an hour
                monitor_posts(subreddit_name, post_limit=1)
        
        logging.info(f"Finished processing r/{subreddit_name}")
        time.sleep(300)  # Wait 5 minutes before processing the next subreddit

if __name__ == "__main__":
    main()
