#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

#Variables that contains the user credentials to access Twitter API
access_token = '793664020456755200-JuOf9BRY9hrGQGcUo9YrQPbECkhf5uC'
access_token_secret = 'FHeTcGvzdW9k3jthSvGNpIXGjRnYQioSWYl43PcbYmIQZ'
consumer_key = 'EAWv8ARtjJ9oCqgeV7Pc9viWy'
consumer_secret = '2TXgZOMmNx2oNeANcRuUEjvdcqM6ODqWJYIuXCRaxq1dR1SFw9'


#This is a basic listener
class StdOutListener(StreamListener):

    def on_data(self, data):
        print data
        return True

    def on_error(self, status):
        print status


if __name__ == '__main__':

    #This handles Twitter authentication and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    #This line filter Twitter Streams to capture data by the keywords
    stream.filter(track=['Pascal Siakam', 'DeMar DeRozan', 'Jonas Valanciunas', 'Kyle Lowry', 'DeMarre Carroll'])
