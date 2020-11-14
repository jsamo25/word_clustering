import simplejson as json
import pandas as pd

from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener

# A twitter API is required, complete the following with your auth keys.
consumer_key = (...)
consumer_secret = (...)
access_token = (...)
access_secret = (...)

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)


class MyListener(StreamListener):
    def __init__(self, api=None):
        super(StreamListener, self).__init__()
        self.num_tweets = 0
    def on_data(self, data):
        try:
            with open('data/MyFile.json', 'a') as f:
                f.write(data)
                twitter_text = json.loads(data)['text']
                print("\n",twitter_text)
                self.num_tweets += 1
                if self.num_tweets < 100:
                    return True
                else:
                    return False
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True
    def on_error(self, status):
        print('Error :', status.place)
        return False

twitter_stream = Stream(auth, MyListener())
twitter_stream.filter(languages=["en"], track=["literature"])  # Add your keywords and other filters

with open('data/MyFile.json') as f:
    data = []
    for line in f:
        try:
            tweets = json.loads(line)
            data.append(tweets["text"])
        except ValueError:
            continue
    data = pd.DataFrame({"text": [x for x in data]})
    print("data size", len(data))
    data.to_csv(r"data/data.csv")

print('[Data Collected]')
print()