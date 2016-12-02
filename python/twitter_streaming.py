#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy import Cursor

#Variables that contains the user credentials to access Twitter API
access_token = '793664020456755200-JuOf9BRY9hrGQGcUo9YrQPbECkhf5uC'
access_token_secret = 'FHeTcGvzdW9k3jthSvGNpIXGjRnYQioSWYl43PcbYmIQZ'
consumer_key = 'EAWv8ARtjJ9oCqgeV7Pc9viWy'
consumer_secret = '2TXgZOMmNx2oNeANcRuUEjvdcqM6ODqWJYIuXCRaxq1dR1SFw9'


#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):

    def on_data(self, data):
        print data
        return True

    def on_error(self, status):
        print status


if __name__ == '__main__':

    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    keyword_list = ['celtics knicks', 'knicks', 'celtics', '']
    for user in Cursor(api.followers, screen_name="celtics").items():
        print user.screen_name
    #This line filter Twitter Streams to capture data by the keywords: 'python', 'javascript', 'ruby'
    # stream.filter(track=keyword_list)
