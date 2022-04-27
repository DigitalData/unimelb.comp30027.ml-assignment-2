import pandas as pd
import numpy as np
from collections import defaultdict
import heapq
import re
import pronouncing as pnc
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# special term regex
RE_LINKS = r"https?://t.co/\w*"
RE_HASHTAGS = r"#\w+"
RE_REFERENCES = r"@\w+"

# define the sets of symbols to construct the emoticons
RE_EMO_EYES = r";:8=" # eye symbols
RE_EMO_MIDDLE = r"\',\-\"\*" # middle symbols
RE_EMO_MOUTHS_HAP = r")3\]" # happy/cutesy mouths
RE_EMO_MOUTHS_HAP_BACK = r"(\[" # if the emote is reversed these are happy mouths
RE_EMO_MOUTHS_SAD = r"\\/(\[" # sad/unhappy mouths
RE_EMO_MOUTHS_SAD_BACK = r")\]" # if the emote is reversed these are sad mouths
RE_EMO_MOUTHS_SUR = r"vo" # surprised mouths
RE_EMO_MOUTHS_NEU = r"pl\|" # neutral mouths
RE_EMO_MOUTHS = RE_EMO_MOUTHS_HAP + RE_EMO_MOUTHS_SAD + RE_EMO_MOUTHS_SUR + RE_EMO_MOUTHS_NEU
# Only allow one type of mouth to be found at a time (`:\3` is not allowed)
RE_EMOTES =  r"(?<=[ ^])[" + RE_EMO_EYES + r"]+[" + RE_EMO_MIDDLE + r"]*[" + RE_EMO_MOUTHS_HAP + r"]+(?=[\W])|"
RE_EMOTES += r"(?<=[ ^])[" + RE_EMO_EYES + r"]+[" + RE_EMO_MIDDLE + r"]*[" + RE_EMO_MOUTHS_SAD + r"]+(?=[\W])|"
RE_EMOTES += r"(?<=[ ^])[" + RE_EMO_EYES + r"]+[" + RE_EMO_MIDDLE + r"]*[" + RE_EMO_MOUTHS_SUR + r"]+(?=[\W])|"
RE_EMOTES += r"(?<=[ ^])[" + RE_EMO_EYES + r"]+[" + RE_EMO_MIDDLE + r"]*[" + RE_EMO_MOUTHS_NEU + r"]+(?=[\W])|"
# add the backwards results
RE_EMOTES += r"(?<=[ ^])[" + RE_EMO_MOUTHS_HAP_BACK + r"]+[" + RE_EMO_MIDDLE + r"]*[" + RE_EMO_EYES + r"]+(?=[\W])|"
RE_EMOTES += r"(?<=[ ^])[" + RE_EMO_MOUTHS_SAD_BACK + r"]+[" + RE_EMO_MIDDLE + r"]*[" + RE_EMO_EYES + r"]+(?=[\W])|"
RE_EMOTES += r"(?<=[ ^])[" + RE_EMO_MOUTHS_NEU + r"]+[" + RE_EMO_MIDDLE + r"]*[" + RE_EMO_EYES + r"]+(?=[\W])"

# Common stop words
STOP_WORDS = ['the', 'th', 'this', 'to', 'you', 'and', 'in', 'is', 'will', 
    'that', 'on', 'of', 'just', 'it', 'for', 'be', 'at', 'about', 'going']
STOP_WORDS = 'english'

# Generating default dictionaries with zero-arrays
def zero_default_dict(tweets):
    def zero_arr(): return np.zeros((len(tweets)))
    return defaultdict(zero_arr)

# add the arrays in a dictionary to a dataframe
def append_dict(tweet_frame, tweet_dict):
    return pd.concat([tweet_frame, pd.DataFrame(tweet_dict)], 
        axis=1)

# takes in a tweet, removes non alpha-numeric symbols and divides into word list
def split_tweet(tweet):
    new_tweet = re.sub(r"-+", " ", tweet) # add spaces instead of - characters (they combine words without spaces)
    new_tweet = re.sub(r" +", "_", new_tweet) # condense repeated spaces into unicode "_"
    new_tweet = re.sub(r"\d+", "_", new_tweet) # remove numbers
    new_tweet = re.sub(r'\W', "", new_tweet) # remove non-alphanumeric symbols [a-zA-Z0-9_]
    new_tweet = re.sub(r"^_||_$", "", new_tweet) # remove the space (_) at the start/end
    return new_tweet.split("_")

# split a tweet into a list of words
def generate_word_lists(tweets):
    return [split_tweet(t) for t in tweets]

# Find the top_n keys in a values dictionary (counts or other values)
def top_n_names(values_dict, top_n = None, sentiments = None, type = 'hybrid'):
    # manage scenarios where nothing needs to be done
    if top_n is None: return values_dict
    if top_n > len(values_dict.keys()): return values_dict

    # define the list of top_n names to extract
    out_names = []

    if sentiments is not None: # determine top_n values per sentiment
        # iterate over the unique values of sentiments
        for s in set(sentiments):
            top_list = []
            for (key, arr) in values_dict.items():
                rating = 0
                if type == 'sum':
                    rating = np.sum(arr[sentiments == s])
                elif type == 'relative':
                    rating = np.sum(arr[sentiments == s]) / np.sum(arr)
                elif type == 'hybrid':
                    rating = np.sum(arr) * np.ma.average(arr[sentiments == s])
                else:
                    rating = np.ma.average(arr[sentiments == s])

                heapq.heappush(top_list, (-1 * rating, key))
                # top_list.append((np.sum(arr[sentiments == s]), key))

            # add the top_n highest values
            out_names += top_list[:top_n]
    else:
        # develop a list of (value, key) tuples where value = sum(dict[key])
        top_list = []
        for (key, arr) in values_dict.items():
            rating = 0
            if type == 'sum':
                rating = np.sum(arr)
            else:
                rating = np.ma.average(arr)
            heapq.heappush(top_list, (-1 * rating, key))
        
        # add the top_n highest values
        out_names += top_list[:top_n]

    # now recreate the dictionary with the smaller list of names
    out_dict = dict()
    for r, name in out_names:
        out_dict[name] = values_dict[name]
    
    return out_dict

################################################################################
### BAG OF WORDS
class TweetBagOfWords(BaseEstimator):
    def __init__(self, top_n = None, by_sentiment = True, stop_words = STOP_WORDS,
            binary = True):
        self.names = []
        self.top_n = top_n
        self.by_sentiment = by_sentiment
        self.sentiments = []
        self.stop_words = stop_words
        self.binary = binary
        self.bow = CountVectorizer(stop_words=self.stop_words, binary=self.binary, 
            tokenizer=split_tweet)

    def get_params(self, deep=True):
        return {
            'top_n': self.top_n,
            'by_sentiment': self.by_sentiment,
            'stop_words': self.stop_words,
        }

    def fit(self, tweets, sentiments=None, **kwargs): return self

    def fit_transform(self, tweets, sentiments=None, **kwargs):
        self.sentiments = sentiments
        if self.top_n is None:
            return self.bow.fit_transform(tweets)

        # otherwise find the top_n
        out_dict = self.generate_dict(tweets)
        self.names = out_dict.keys()
        self.bow = CountVectorizer(stop_words = self.stop_words, 
            vocabulary = self.names)
        return self.bow.fit_transform(tweets)

    def transform(self, tweets, sentiments=None, **kwargs):
        return self.bow.transform(tweets)

    def generate_dict(self, tweets):
        train_bow_matrix = self.bow.fit_transform(tweets)

        # create the BoW dictionary
        dict_bow = zero_default_dict(tweets)
        bow_words = self.bow.get_feature_names_out()
        for id1, row in enumerate(train_bow_matrix):
            for id2, val in zip(row.indices, row.data):
                dict_bow[bow_words[id2]][id1] = val / len(row.data)

        if self.by_sentiment:
            return top_n_names(dict_bow, self.top_n, sentiments=self.sentiments)
        else:
            return top_n_names(dict_bow, self.top_n)


################################################################################
### TF-IDF
class TweetTFIDF(BaseEstimator):
    def __init__(self, top_n = None, by_sentiment = True, stop_words = STOP_WORDS):
        self.names = []
        self.top_n = top_n
        self.by_sentiment = by_sentiment
        self.sentiments = []
        self.stop_words = stop_words
        self.tfidf = TfidfVectorizer(stop_words = self.stop_words, 
            tokenizer=split_tweet)

    def get_params(self, deep=True):
        return {
            'top_n': self.top_n,
            'by_sentiment': self.by_sentiment,
            'stop_words': self.stop_words,
        }

    def fit(self, tweets, sentiments=None, **kwargs): return self

    def fit_transform(self, tweets, sentiments=None, **kwargs):
        self.sentiments = sentiments
        if self.top_n is None:
            return self.tfidf.fit_transform(tweets)

        # otherwise find the top_n
        out_dict = self.generate_dict(tweets)
        self.names = out_dict.keys()
        self.tfidf = TfidfVectorizer(stop_words = self.stop_words, 
            vocabulary = self.names)
        return self.tfidf.fit_transform(tweets)

    def transform(self, tweets, sentiments=None, **kwargs):
        return self.tfidf.transform(tweets)

    def generate_dict(self, tweets):
        train_tfidf_matrix = self.tfidf.fit_transform(tweets)

        # create the tfidf dictionary
        dict_tfidf = zero_default_dict(tweets)
        tfidf_words = self.tfidf.get_feature_names_out()
        for id1, row in enumerate(train_tfidf_matrix):
            for id2, val in zip(row.indices, row.data):
                dict_tfidf[tfidf_words[id2]][id1] = val / len(row.data)

        if self.by_sentiment:
            return top_n_names(dict_tfidf, self.top_n, sentiments=self.sentiments)
        else:
            return top_n_names(dict_tfidf, self.top_n)


################################################################################
### NUMERIC METRICS
class TweetMetrics(BaseEstimator):
    
    def __init__(self):
        self.names = ['word', 'char', 'alphanumeric', 'alphabetic', 
            'links', 'hashtags', 'references', 'emotes']

    def get_params(self, deep=True):
        return {'names': self.names}

    def fit(self, tweets, sentiments=None, **kwargs): return self

    def fit_transform(self, tweets, sentiments=None, **kwargs):
        output = np.zeros((len(tweets), len(self.names)))
        for idx, tweet in enumerate(tweets):
            words = split_tweet(tweet)
            output[idx, 0] = len(words)
            output[idx, 1] = len(tweet)
            tweet_alphanum = re.sub(r"[\W_]", "", tweet)
            output[idx, 2] = len(tweet_alphanum)
            tweet_alpha = re.sub(r"\d", "", tweet_alphanum)
            output[idx, 3] = len(tweet_alpha)
            output[idx, 4] = len(re.findall(RE_LINKS, tweet))
            output[idx, 5] = len(re.findall(RE_HASHTAGS, tweet))
            output[idx, 6] = len(re.findall(RE_REFERENCES, tweet))
            output[idx, 7] = len(re.findall(RE_EMOTES, tweet))
        return output

    def transform(self, tweets, sentiments=None, **kwargs):
        return self.fit_transform(tweets)

    # Create the dictionary of different length metrics
    def generate_dict(self, tweets):
        # extract tweets and words
        word_lists = generate_word_lists(tweets)

        # create a default dict
        dict_character_counts = zero_default_dict(tweets)

        for idx, (tweet, words) in enumerate(zip(tweets, word_lists)):
            dict_character_counts['word'][idx] = len(words)
            dict_character_counts['char'][idx] = len(tweet)
            tweet_alphanum = re.sub(r"[\W_]", "", tweet)
            dict_character_counts['alnum'][idx] = len(tweet_alphanum)
            tweet_alpha = re.sub(r"\d", "", tweet_alphanum)
            dict_character_counts['alpha'][idx] = len(tweet_alpha)
            dict_character_counts['links'][idx] = len(re.findall(RE_LINKS, tweet))
            dict_character_counts['hashtags'][idx] = len(re.findall(RE_HASHTAGS, tweet))
            dict_character_counts['references'][idx] = len(re.findall(RE_REFERENCES, tweet))
            dict_character_counts['emotes'][idx] = len(re.findall(RE_EMOTES, tweet))
        return dict_character_counts


################################################################################
### WORD LENGTH DISTRIBUTIONS
class TweetWordLengths(BaseEstimator):
    def __init__(self, top_n = None, by_sentiment = True, relative = False):
        self.names = []
        self.top_n = top_n
        self.by_sentiment = by_sentiment
        self.sentiments = []
        self.relative = relative

    def get_params(self, deep=True):
        return {
            'top_n': self.top_n,
            'by_sentiment': self.by_sentiment,
            'relative': self.relative,
        }

    def fit(self, tweets, sentiments=None, **kwargs): return self

    def fit_transform(self, tweets, sentiments=None, **kwargs):
        self.sentiments = sentiments
        dict_wl = self.generate_dict(tweets)
        numbers = dict_wl.keys()
        self.names = numbers
        output = np.zeros((len(tweets), len(self.names)))
        for id1, name in enumerate(self.names):
            for id2, num in enumerate(dict_wl[name]):
                output[id2, id1] = num
        return output

    def transform(self, tweets, sentiments=None, **kwargs):
        output = np.zeros((len(tweets), len(self.names)))
        dict_wl = self.generate_dict(tweets)
        for id1, name in enumerate(self.names):
            for id2, num in enumerate(dict_wl[name]):
                output[id2, id1] = num
        return output

    # define a breakdown of the lengths of words
    def generate_dict(self, tweets):
        # extract tweets and words
        word_lists = generate_word_lists(tweets)

        # create a default dict
        dict_word_lengths = zero_default_dict(tweets)

        for idx, wl in enumerate(word_lists):
            if self.relative:
                for w in wl:
                    dict_word_lengths[len(w)][idx] += 1 / len(wl)
            else:
                for w in wl:
                    dict_word_lengths[len(w)][idx] += 1

        if self.by_sentiment:
            return top_n_names(dict_word_lengths, self.top_n, sentiments=self.sentiments)
        else:
            return top_n_names(dict_word_lengths, self.top_n)


################################################################################
### CHARACTER FREQUENCIES
class TweetCharacterFrequencies(BaseEstimator):
    
    def __init__(self, top_n = None, by_sentiment = True, relative = False, alphabetic = True):
        self.names = []
        self.top_n = top_n
        self.by_sentiment = by_sentiment
        self.sentiments = []
        self.relative = relative
        self.alphabetic = alphabetic

    def get_params(self, deep=True):
        return {
            'top_n': self.top_n,
            'by_sentiment': self.by_sentiment,
            'relative': self.relative,
            'alphabetic': self.alphabetic,
        }

    def fit(self, tweets, sentiments=None, **kwargs): return self

    def fit_transform(self, tweets, sentiments=None, **kwargs):
        self.sentiments = sentiments
        dict_cf = self.generate_dict(tweets)
        self.names = list(dict_cf.keys())
        output = np.zeros((len(tweets), len(self.names)))

        for id1, (key, arr) in enumerate(dict_cf.items()):
            for id2, num in enumerate(arr):
                output[id2, id1] = num

        return output

    def transform(self, tweets, sentiments=None, **kwargs):
        dict_cf = self.generate_dict(tweets)
        output = np.zeros((len(tweets), len(self.names)))

        for id1, name in enumerate(self.names):
            for id2, num in enumerate(dict_cf[name]):
                output[id2, id1] = num

        return output

    # define a dictionary of arrays of relative/absolute frequencies per tweet per character
    def generate_dict(self, tweets):
        # create a default dict
        dict_character_freqs = zero_default_dict(tweets)

        # iterate through the tweets
        for idx, t in enumerate(tweets):

            # isolate only alphabetics
            if self.alphabetic:
                t = re.sub(r"\W", "", t)
                t = re.sub(r"[\d_]", "", t)

            # iterate through the chars in t and add to their counts
            # then divide by the total length of the tweet (if relative)
            if self.relative:
                for c in t:
                    dict_character_freqs[c][idx] += 1 / len(t)
            else:
                for c in t:
                    dict_character_freqs[c][idx] += 1

        if self.by_sentiment:
            return top_n_names(dict_character_freqs, self.top_n, sentiments=self.sentiments)
        else:
            return top_n_names(dict_character_freqs, self.top_n)


################################################################################
### LINKS
class TweetLinks(BaseEstimator):
    def __init__(self, top_n = None, by_sentiment = True,
            relative = False):
        self.names = []
        self.top_n = top_n
        self.by_sentiment = by_sentiment
        self.sentiments = []
        self.relative = relative

    def get_params(self, deep=True):
        return {
            'top_n': self.top_n,
            'by_sentiment': self.by_sentiment,
            'relative': self.relative,
        }

    def fit(self, tweets, sentiments=None, **kwargs): return self

    def fit_transform(self, tweets, sentiments=None, **kwargs):
        self.sentiments = sentiments
        dict_l_counts = self.generate_dict(tweets)
        self.names = list(dict_l_counts.keys())
        output = np.zeros((len(tweets), len(self.names)))

        for id1, (key, arr) in enumerate(dict_l_counts.items()):
            for id2, num in enumerate(arr):
                output[id2, id1] = num

        return output

    def transform(self, tweets, sentiments=None, **kwargs):
        dict_l_counts = self.generate_dict(tweets)
        output = np.zeros((len(tweets), len(self.names)))

        for id1, name in enumerate(self.names):
            for id2, num in enumerate(dict_l_counts[name]):
                output[id2, id1] = num

        return output

    # create a list of links in a tweet
    def generate_dict(self, tweets):
        # create a default dict for occurrences of specific links, as well as general link counts
        dict_link_counts = zero_default_dict(tweets)

        for idx, t in enumerate(tweets):
            links = re.findall(RE_LINKS, t)

            # iterate through the refs in t and add to their counts
            # then divide by the total number of refs of the tweet (if relative)
            if self.relative:
                for l in links: 
                    dict_link_counts[l][idx] += 1 / len(links)
            else:
                for l in links:
                    dict_link_counts[l][idx] += 1

        if self.by_sentiment:
            return top_n_names(dict_link_counts, self.top_n, sentiments=self.sentiments)
        else:
            return top_n_names(dict_link_counts, self.top_n)


################################################################################
### HASHTAGS
class TweetHashtags(BaseEstimator):
    def __init__(self, top_n = None, by_sentiment = True,
            relative = False):
        self.names = []
        self.top_n = top_n
        self.by_sentiment = by_sentiment
        self.sentiments = []
        self.relative = relative

    def get_params(self, deep=True):
        return {
            'top_n': self.top_n,
            'by_sentiment': self.by_sentiment,
            'relative': self.relative,
        }

    def fit(self, tweets, sentiments=None, **kwargs): return self

    def fit_transform(self, tweets, sentiments=None, **kwargs):
        self.sentiments = sentiments
        dict_h_counts = self.generate_dict(tweets)
        self.names = list(dict_h_counts.keys())
        output = np.zeros((len(tweets), len(self.names)))

        for id1, (key, arr) in enumerate(dict_h_counts.items()):
            for id2, num in enumerate(arr):
                output[id2, id1] = num

        return output

    def transform(self, tweets, sentiments=None, **kwargs):
        dict_h_counts = self.generate_dict(tweets)
        output = np.zeros((len(tweets), len(self.names)))

        for id1, name in enumerate(self.names):
            for id2, num in enumerate(dict_h_counts[name]):
                output[id2, id1] = num

        return output

    # create a list of hashtags in a tweet
    def generate_dict(self, tweets):
        # create a default dict for occurrences of specific hashtags, as well as general hashtag counts
        dict_hashtag_counts = zero_default_dict(tweets)

        for idx, t in enumerate(tweets):
            hashes = re.findall(RE_HASHTAGS, t)

            # iterate through the refs in t and add to their counts
            # then divide by the total number of refs of the tweet (if relative)
            if self.relative:
                for h in hashes: 
                    dict_hashtag_counts[h][idx] += 1 / len(hashes)
            else:
                for h in hashes:
                    dict_hashtag_counts[h][idx] += 1

        if self.by_sentiment:
            return top_n_names(dict_hashtag_counts, self.top_n, sentiments=self.sentiments)
        else:
            return top_n_names(dict_hashtag_counts, self.top_n)


################################################################################
### REFERENCES
class TweetReferences(BaseEstimator):
    def __init__(self, top_n = None, by_sentiment = True,
            relative = False):
        self.names = []
        self.top_n = top_n
        self.by_sentiment = by_sentiment
        self.sentiments = []
        self.relative = relative

    def get_params(self, deep=True):
        return {
            'top_n': self.top_n,
            'by_sentiment': self.by_sentiment,
            'relative': self.relative,
        }

    def fit(self, tweets, sentiments=None, **kwargs): return self

    def fit_transform(self, tweets, sentiments=None, **kwargs):
        self.sentiments = sentiments
        dict_r_counts = self.generate_dict(tweets)
        self.names = list(dict_r_counts.keys())
        output = np.zeros((len(tweets), len(self.names)))

        for id1, (key, arr) in enumerate(dict_r_counts.items()):
            for id2, num in enumerate(arr):
                output[id2, id1] = num

        return output

    def transform(self, tweets, sentiments=None, **kwargs):
        dict_r_counts = self.generate_dict(tweets)
        output = np.zeros((len(tweets), len(self.names)))

        for id1, name in enumerate(self.names):
            for id2, num in enumerate(dict_r_counts[name]):
                output[id2, id1] = num

        return output

    # create a list of user references in a tweet
    def generate_dict(self, tweets):
        # create a default dict for occurrences of specific references, as well as general reference counts
        dict_reference_counts = zero_default_dict(tweets)

        for idx, t in enumerate(tweets):
            refs = re.findall(RE_REFERENCES, t)

            # iterate through the refs in t and add to their counts
            # then divide by the total number of refs of the tweet (if relative)
            if self.relative:
                for r in refs: 
                    dict_reference_counts[r][idx] += 1 / len(refs)
            else:
                for r in refs:
                    dict_reference_counts[r][idx] += 1

        if self.by_sentiment:
            return top_n_names(dict_reference_counts, self.top_n, sentiments=self.sentiments)
        else:
            return top_n_names(dict_reference_counts, self.top_n)


################################################################################
### EMOTICONS
class TweetEmoticons(BaseEstimator):
    def __init__(self, top_n = None, by_sentiment = True,
            relative = False,
            simplify = True):
        self.names = []
        self.top_n = top_n
        self.by_sentiment = by_sentiment
        self.sentiments = []
        self.relative = relative
        self.simplify = simplify

    def get_params(self, deep=True):
        return {
            'top_n': self.top_n,
            'by_sentiment': self.by_sentiment,
            'relative': self.relative,
            'simplify': self.simplify,
        }

    def fit(self, tweets, sentiments=None, **kwargs): return self

    def fit_transform(self, tweets, sentiments=None, **kwargs):
        self.sentiments = sentiments
        dict_e_counts = self.generate_dict(tweets)
        self.names = list(dict_e_counts.keys())
        output = np.zeros((len(tweets), len(self.names)))

        for id1, (key, arr) in enumerate(dict_e_counts.items()):
            for id2, num in enumerate(arr):
                output[id2, id1] = num

        return output

    def transform(self, tweets, sentiments=None, **kwargs):
        dict_e_counts = self.generate_dict(tweets)
        output = np.zeros((len(tweets), len(self.names)))

        for id1, name in enumerate(self.names):
            for id2, num in enumerate(dict_e_counts[name]):
                output[id2, id1] = num
        return output 

    # create a breakdown of the emoticons used in the tweets
    def generate_dict(self, tweets):
        # create the dictionaries of emotes (exhaustive and simpliefied) and their occurrence lists over the tweets
        dict_emoticon_counts = zero_default_dict(tweets);

        # Simplifies emoticons to remove middle symbols and simplify eyes 
        # (since it's really the mouth that gives emotion away)
        def simplify_emoticon(emote):
            DEFAULT_EYES = ":"
            DEFAULT_NOSE = ""
            DEFAULT_MOUTH_HAP = ")"
            DEFAULT_MOUTH_SAD = "("
            DEFAULT_MOUTH_SUR = "o"
            DEFAULT_MOUTH_NEU = "|"

            # reverse it if needed
            if emote[0] in RE_EMO_MOUTHS_HAP_BACK:
                emote = emote[::-1]
                emote = re.sub(r"[" + RE_EMO_MOUTHS_HAP_BACK + r"]+", 
                    DEFAULT_MOUTH_HAP, emote)
            elif emote[0] in RE_EMO_MOUTHS_SAD_BACK:
                emote = emote[::-1]
                emote = re.sub(r"[" + RE_EMO_MOUTHS_SAD_BACK + r"]+", 
                    DEFAULT_MOUTH_SAD, emote)
            elif emote[0] in RE_EMO_MOUTHS_NEU:
                emote = emote[::-1]

            # shrink the emote to one of the simple :) :( :| :o emotes.
            e_simple = ""
            for symbol in emote:
                if symbol in RE_EMO_EYES:
                    symbol = DEFAULT_EYES
                elif symbol in RE_EMO_MIDDLE: 
                    symbol = DEFAULT_NOSE
                elif symbol in RE_EMO_MOUTHS_HAP:
                    symbol = DEFAULT_MOUTH_HAP
                elif symbol in RE_EMO_MOUTHS_SAD:
                    symbol = DEFAULT_MOUTH_SAD
                elif symbol in RE_EMO_MOUTHS_SUR:
                    symbol = DEFAULT_MOUTH_SUR
                elif symbol in RE_EMO_MOUTHS_NEU:
                    symbol = DEFAULT_MOUTH_NEU
                if len(e_simple) > 0:
                    if symbol == e_simple[-1]: continue
                    if symbol in RE_EMO_MOUTHS and e_simple[-1] in RE_EMO_MOUTHS:
                        continue
                e_simple += symbol

            return e_simple

        # iterate through tweets and extract the emoticons
        for idx, t in enumerate(tweets):
            # extract valid emotes
            emotes = re.findall(RE_EMOTES, t)
            def isvalid(e): # remove purely alphanumeric emotes
                return not (e.isalnum() or e.isdecimal() or e.isdigit())
            emotes = list(filter(isvalid, emotes))
            
            # iterate through the emotes in this tweet and append their occurrence counts
            if self.relative:
                for e in emotes:
                    if self.simplify:
                        dict_emoticon_counts[simplify_emoticon(e)][idx] += 1 / len(emotes)
                    else:
                        dict_emoticon_counts[e][idx] += 1 / len(emotes)
            else:
                for e in emotes:
                    if self.simplify:
                        dict_emoticon_counts[simplify_emoticon(e)][idx] += 1
                    else:
                        dict_emoticon_counts[e][idx] += 1

        if self.by_sentiment:
            return top_n_names(dict_emoticon_counts, self.top_n, sentiments=self.sentiments)
        else:
            return top_n_names(dict_emoticon_counts, self.top_n)


################################################################################
### PHONETICS
class TweetPhonetics(BaseEstimator):
    def __init__(self, top_n = None, by_sentiment = True, relative = False):
        self.names = []
        self.top_n = top_n
        self.by_sentiment = by_sentiment
        self.sentiments = []
        self.relative = relative

    def get_params(self, deep=True):
        return {
            'top_n': self.top_n,
            'by_sentiment': self.by_sentiment,
            'relative': self.relative,
        }

    def fit(self, tweets, sentiments=None, **kwargs): return self

    def fit_transform(self, tweets, sentiments=None, **kwargs):
        self.sentiments = sentiments
        dict_p_counts = self.generate_dict(tweets)
        self.names = list(dict_p_counts.keys())
        output = np.zeros((len(tweets), len(self.names)))

        for id1, (key, arr) in enumerate(dict_p_counts.items()):
            for id2, num in enumerate(arr):
                output[id2, id1] = num

        return output

    def transform(self, tweets, sentiments=None, **kwargs):
        dict_p_counts = self.generate_dict(tweets)
        output = np.zeros((len(tweets), len(self.names)))

        for id1, name in enumerate(self.names):
            for id2, num in enumerate(dict_p_counts[name]):
                output[id2, id1] = num
        return output 

    # create a breakdown of the phonetics used in the tweets
    def generate_dict(self, tweets):
        # create the dictionaries of emotes (exhaustive and simpliefied) and their occurrence lists over the tweets
        dict_phonetic_counts = zero_default_dict(tweets);

        # break the tweets into word lists
        word_lists = generate_word_lists(tweets)

        # iterate through tweets and extract the emoticons
        for idx, words in enumerate(word_lists):
            # iterate through the words
            for w in words:
                # Break the word into its phones
                phones = pnc.phones_for_word(w)

                # break the phones into a list
                if len(phones) > 0:
                    phones = phones[0].split(' ')
                
                # iterate through the phones and add to the counts
                if self.relative:
                    for p in phones:
                        dict_phonetic_counts[p][idx] += 1 / len(words)
                else:
                    for p in phones:
                        dict_phonetic_counts[p][idx] += 1

        if self.by_sentiment:
            return top_n_names(dict_phonetic_counts, self.top_n, sentiments=self.sentiments)
        else:
            return top_n_names(dict_phonetic_counts, self.top_n)


################################################################################
### POETIC PHONETICS
class TweetPoetics(BaseEstimator):
    def __init__(self, relative = False):
        self.names = ['plosive', 'sibilant', 'fricative', 
            'fricative-hard', 'fricative-soft']
        self.relative = relative

    def get_params(self, deep=True):
        return {
            'names': self.names,
            'relative': self.relative,
        }

    def fit(self, tweets, sentiments=None, **kwargs): return self

    def fit_transform(self, tweets, sentiments=None, **kwargs):
        dict_p_counts = self.generate_dict(tweets)
        self.names = list(dict_p_counts.keys())
        output = np.zeros((len(tweets), len(self.names)))

        for id1, (key, arr) in enumerate(dict_p_counts.items()):
            for id2, num in enumerate(arr):
                output[id2, id1] = num

        return output

    def transform(self, tweets, sentiments=None, **kwargs):
        dict_p_counts = self.generate_dict(tweets)
        output = np.zeros((len(tweets), len(self.names)))

        for id1, name in enumerate(self.names):
            for id2, num in enumerate(dict_p_counts[name]):
                output[id2, id1] = num
        return output 

    # create a breakdown of the phonetics used in the tweets
    def generate_dict(self, tweets):
        # create the dictionaries of emotes (exhaustive and simpliefied) and their occurrence lists over the tweets
        dict_poetic_counts = zero_default_dict(tweets);

        # sets of poetic noises
        # using: https://nlp.stanford.edu/courses/lsa352/arpabet.html 
        plosive_set = ['B', 'P', 'D', 'T', 'G', 'K', 'BCL', 'PCL', 'DCL', 'TCL', 'GCL', 'KCL', 'DX']
        re_plosive = r"(?<=[ ^])" + r"(?=[ $])|(?<=[ ^])".join(plosive_set) + r"(?=[ $])"
        fricative_hard_set = ['DH', 'V']
        re_fricative_hard = r"(?<=[ ^])" + r"(?=[ $])|(?<=[ ^])".join(fricative_hard_set) + r"(?=[ $])"
        fricative_soft_set = ['TH', 'F']
        re_fricative_soft = r"(?<=[ ^])" + r"(?=[ $])|(?<=[ ^])".join(fricative_soft_set) + r"(?=[ $])"
        sibilant_set = ['Z', 'S', 'CH', 'SH', 'ZH']
        re_sibilant = r"(?<=[ ^])" + r"(?=[ $])|(?<=[ ^])".join(sibilant_set) + r"(?=[ $])"

        # break the tweets into word lists
        word_lists = generate_word_lists(tweets)

        # iterate through tweets and extract the emoticons
        for idx, words in enumerate(word_lists):
            # iterate through the words
            for w in words:
                # Break the word into its phones
                phones = " ".join(pnc.phones_for_word(w))
                
                if re.search(re_plosive, phones) is not None:
                    dict_poetic_counts["plosive"][idx] += 1
                
                if re.search(re_sibilant, phones) is not None:
                    dict_poetic_counts["sibilant"][idx] += 1

                if re.search(re_fricative_hard, phones) is not None:
                    dict_poetic_counts["fricative"][idx] += 1
                    dict_poetic_counts["fricative-hard"][idx] += 1

                if re.search(re_fricative_soft, phones) is not None:
                    dict_poetic_counts["fricative"][idx] += 1
                    dict_poetic_counts["fricative-soft"][idx] += 1
                    
        return dict_poetic_counts