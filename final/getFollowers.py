import tweepy
from tweepy import OAuthHandler
from tweepy import Cursor
from tweepy import API

access_token = '793664020456755200-JuOf9BRY9hrGQGcUo9YrQPbECkhf5uC'
access_token_secret = 'FHeTcGvzdW9k3jthSvGNpIXGjRnYQioSWYl43PcbYmIQZ'
consumer_key = 'EAWv8ARtjJ9oCqgeV7Pc9viWy'
consumer_secret = '2TXgZOMmNx2oNeANcRuUEjvdcqM6ODqWJYIuXCRaxq1dR1SFw9'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit = True, compression = True)

ids = []
for user in tweepy.Cursor(api.followers, screen_name="celtics").items():
    print user.screen_name
