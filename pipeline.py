import datetime
import  pandas as pd
from data_scrappers import reddit_scraper, reddit_post_scrapper, twitter_scraper_
from data_cleaning import data_cleaning
from Sentiment_analysis import vader_model



# scrapes the most popular tickers from wsb
wsb_tickers = reddit_scraper.get_tickers()
wsb_tickers = wsb_tickers.sort_values(ascending=False)
top_15_tickers = wsb_tickers.head(15).index
print('Finished Scrapping Most Popular Tickers on WSB. Now Scrapping the the WSB titles')
# scrapes titles from popular tickers on wbs
wsb_titles = reddit_post_scrapper.scrape_posts(top_15_tickers)

# scrapes tweets that contain popular tickers from wsb
print("Finished Scrapping wsb titles. Now scrapping twitter")
tweets = twitter_scraper_.get_tweets(top_15_tickers)

print('Cleaning tweets')
# cleans tweets
tweets.columns = ['Date', 'Ticker', 'Tweet']
tweets['Tweet'] = tweets['Tweet'].map(lambda x: data_cleaning.cleaner(x))
tweets['Date'] = pd.to_datetime(tweets['Date']).dt.date

print('Cleaning titles from WSB')
# cleans titles from wsb
wsb_titles.columns = ['Ticker', 'Title', 'Date']
wsb_titles['Title'] = wsb_titles['Title'].map(lambda x: data_cleaning.cleaner(x))
wsb_titles['Date'] = pd.to_datetime(wsb_titles['Date'], utc=True, unit='s').dt.date

tweets = tweets.dropna()
wsb_titles = wsb_titles.dropna()

print("Running tweets and wsb titles through vader model")
# runs the vader sentiment model on the tweets and the wsb titles
twitter_sentiments = vader_model.sentiment_df(tweets, 'Tweet')
wsb_sentiments = vader_model.sentiment_df(wsb_titles, 'Title')

twitter_sentiments.to_csv('tweet_sentiments.csv', index=False)
wsb_sentiments.to_csv('wsb_titles_sentiments.csv', index=False)

print('Done')