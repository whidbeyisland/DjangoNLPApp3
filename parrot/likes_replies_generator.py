import random
import math

class LikesRepliesGenerator:
    num_likes_str = '50'
    num_replies_str = '6'
    strs_to_return = []

    def __init__(self):
        pass

    def generate(self, num_tweets):
        for i in range(0, num_tweets):
            magnitude = random.randint(0, 5)
            num_likes = (int)(random.uniform(1, 9.99999) * math.pow(10, magnitude))
            ratio = random.uniform(0.4, 14)
            num_replies = (int)(num_likes / ratio)

            num_likes_str = str(num_likes) if num_likes < 1000 else str(num_likes)[:-3] + 'K'
            num_replies_str = str(num_replies) if num_replies < 1000 else str(num_replies)[:-3] + 'K'
            self.strs_to_return.append([num_likes_str, num_replies_str])
        return self.strs_to_return