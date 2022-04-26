import enum
import pandas as pd
import numpy as np
from collections import defaultdict
import re
import pronouncing as pnc
from scipy.sparse import csr_matrix

RE_LINKS = r"https://t.co/\S*"
RE_HASHTAGS = r"#\w+"
RE_REFERENCES = r"@\w+"

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
    new_tweet = re.sub(r'\W', "", new_tweet) # remove non-alphanumeric symbols [a-zA-Z0-9_]
    new_tweet = re.sub(r"^_||_$", "", new_tweet) # remove the space (_) at the start/end
    return new_tweet.split("_")

# split a tweet into a list of words
def generate_word_list(tweets):
    return [split_tweet(t) for t in tweets]

################################################################################
### NUMERIC METRICS
class TweetMetrics():
    
    def __init__(self):
        self.names = ['word', 'char', 'alphanumeric', 'alphabetic', 'links', 'hashtags', 'references']

    def fit_transform(self, tweets):
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
        return output

    def transform(self, tweets):
        return self.fit_transform(tweets)

# Create the dataframe of different length metrics
def generate_lengths_dict(tweets):
    # extract tweets and words
    word_lists = generate_word_list(tweets)

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
    return dict_character_counts


################################################################################
### WORD LENGTH DISTRIBUTIONS
class TweetWordLengths():
    def __init__(self, relative = True):
        self.names = []
        self.relative = relative
        self.max = 1
        self.min = 0

    def fit_transform(self, tweets):
        dict_wl = generate_word_lengths_dict(tweets, relative = self.relative)
        numbers = dict_wl.keys()
        self.max = max(numbers)
        self.names = range(self.min, self.max)
        output = np.zeros((len(tweets), 1 + self.max))
        for key, arr in dict_wl.items():
            for idx, num in enumerate(arr):
                output[idx, key] = num
        return output

    def transform(self, tweets):
        output = np.zeros((len(tweets), 1 + self.max))
        dict_wl = generate_word_lengths_dict(tweets, relative = self.relative)
        for key, arr in dict_wl.items():
            for idx, num in enumerate(arr):
                output[idx, key] = num
        return output


# define a breakdown of the lengths of words
def generate_word_lengths_dict(tweets, relative = True):
    # extract tweets and words
    word_lists = generate_word_list(tweets)

    # create a default dict
    dict_word_lengths = zero_default_dict(tweets)

    for idx, wl in enumerate(word_lists):
        if relative:
            for w in wl:
                dict_word_lengths[len(w)][idx] += 1 / len(wl)
        else:
            for w in wl:
                dict_word_lengths[len(w)][idx] += 1

    return dict_word_lengths


################################################################################
### CHARACTER FREQUENCIES
class TweetCharacterFrequencies():
    
    def __init__(self, relative = True):
        self.names = []
        self.relative = relative

    def fit_transform(self, tweets):
        dict_cf = generate_char_freq_dict(tweets, relative = self.relative)
        self.names = dict_cf.keys()
        output = np.zeros((len(tweets), len(self.names)))

        for id1, (key, arr) in enumerate(dict_cf.items()):
            for id2, num in enumerate(arr):
                output[id2, id1] = num

        return output

    def transform(self, tweets):
        dict_cf = generate_char_freq_dict(tweets, relative = self.relative)
        output = np.zeros((len(tweets), len(self.names)))

        for id1, name in enumerate(self.names):
            for id2, num in enumerate(dict_cf[name]):
                output[id2, id1] = num

        return output


# define a dictionary of arrays of relative/absolute frequencies per tweet per character
def generate_char_freq_dict(tweets, relative = True, alphabetic = False):
    # create a default dict
    dict_character_freqs = zero_default_dict(tweets)

    # iterate through the tweets
    for idx, t in enumerate(tweets):

        # isolate only alphabetics
        if alphabetic:
            t = re.sub(r"\W", "", t)
            t = re.sub(r"[\d_]", "", t)

        # iterate through the chars in t and add to their counts
        # then divide by the total length of the tweet (if relative)
        if relative:
            for c in t:
                dict_character_freqs[c][idx] += 1 / len(t)
        else:
            for c in t:
                dict_character_freqs[c][idx] += 1

    return dict_character_freqs


################################################################################
### LINKS
class TweetLinks():
    def __init__(self,
            relative = True):
        self.names = []
        self.relative = relative

    def fit_transform(self, tweets):
        dict_l, dict_l_counts = generate_links(tweets, relative = self.relative)
        self.names = dict_l_counts.keys()
        output = np.zeros((len(tweets), len(self.names)))

        for id1, (key, arr) in enumerate(dict_l_counts.items()):
            for id2, num in enumerate(arr):
                output[id2, id1] = num

        return output

    def transform(self, tweets):
        dict_l, dict_l_counts = generate_links(tweets, relative = self.relative)
        output = np.zeros((len(tweets), len(self.names)))

        for id1, name in enumerate(self.names):
            for id2, num in enumerate(dict_l_counts[name]):
                output[id2, id1] = num

        return output


# create a list of links in a tweet
def generate_links(tweets, relative = True, counts = True):
    # create a default dict for occurrences of specific links, as well as general link counts
    dict_link_counts = zero_default_dict(tweets)
    dict_links = zero_default_dict(tweets)

    # the list of lists of references per tweet is not an ndarray (set by `zero_default_dict`)
    dict_links['values'] = []

    for idx, t in enumerate(tweets):
        links = re.findall(RE_LINKS, t)
        dict_links['num'][idx] = len(links)
        dict_links['values'].append(links)

        # iterate through the refs in t and add to their counts
        # then divide by the total number of refs of the tweet (if relative)
        if counts:
            if relative:
                for l in links: 
                    dict_link_counts[l][idx] += 1 / len(links)
            else:
                for l in links:
                    dict_link_counts[l][idx] += 1

    dict_links['num'] = dict_links['num'].astype(int)
    return (dict_links, dict_link_counts)

################################################################################
### HASHTAGS
class TweetHashtags():
    def __init__(self,
            relative = True):
        self.names = []
        self.relative = relative

    def fit_transform(self, tweets):
        dict_h, dict_h_counts = generate_hashtags(tweets, relative = self.relative)
        self.names = dict_h_counts.keys()
        output = np.zeros((len(tweets), len(self.names)))

        for id1, (key, arr) in enumerate(dict_h_counts.items()):
            for id2, num in enumerate(arr):
                output[id2, id1] = num

        return output

    def transform(self, tweets):
        dict_h, dict_h_counts = generate_hashtags(tweets, relative = self.relative)
        output = np.zeros((len(tweets), len(self.names)))

        for id1, name in enumerate(self.names):
            for id2, num in enumerate(dict_h_counts[name]):
                output[id2, id1] = num

        return output

# create a list of hashtags in a tweet
def generate_hashtags(tweets, relative = True, counts = True):
    # create a default dict for occurrences of specific hashtags, as well as general hashtag counts
    dict_hashtag_counts = zero_default_dict(tweets)
    dict_hashtags = zero_default_dict(tweets)

    # the list of lists of references per tweet is not an ndarray (set by `zero_default_dict`)
    dict_hashtags['values'] = []

    for idx, t in enumerate(tweets):
        hashes = re.findall(RE_HASHTAGS, t)
        dict_hashtags['num'][idx] = len(hashes)
        dict_hashtags['values'].append(hashes)

        # iterate through the refs in t and add to their counts
        # then divide by the total number of refs of the tweet (if relative)
        if counts:
            if relative:
                for h in hashes: 
                    dict_hashtag_counts[h][idx] += 1 / len(hashes)
            else:
                for h in hashes:
                    dict_hashtag_counts[h][idx] += 1

    dict_hashtags['num'] = dict_hashtags['num'].astype(int)
    return (dict_hashtags, dict_hashtag_counts)


################################################################################
### REFERENCES
class TweetReferences():
    def __init__(self,
            relative = True):
        self.names = []
        self.relative = relative

    def fit_transform(self, tweets):
        dict_r, dict_r_counts = generate_references(tweets, relative = self.relative)
        self.names = dict_r_counts.keys()
        output = np.zeros((len(tweets), len(self.names)))

        for id1, (key, arr) in enumerate(dict_r_counts.items()):
            for id2, num in enumerate(arr):
                output[id2, id1] = num

        return output

    def transform(self, tweets):
        dict_r, dict_r_counts = generate_references(tweets, relative = self.relative)
        output = np.zeros((len(tweets), len(self.names)))

        for id1, name in enumerate(self.names):
            for id2, num in enumerate(dict_r_counts[name]):
                output[id2, id1] = num

        return output

# create a list of user references in a tweet
def generate_references(tweets, relative = True, counts = True):
    # create a default dict for occurrences of specific references, as well as general reference counts
    dict_reference_counts = zero_default_dict(tweets)
    dict_references = zero_default_dict(tweets)

    # the list of lists of references per tweet is not an ndarray (set by `zero_default_dict`)
    dict_references['users'] = []

    for idx, t in enumerate(tweets):
        refs = re.findall(RE_REFERENCES, t)
        dict_references['num'][idx] = len(refs)
        dict_references['users'].append(refs)

        # iterate through the refs in t and add to their counts
        # then divide by the total number of refs of the tweet (if relative)
        if counts:
            if relative:
                for r in refs: 
                    dict_reference_counts[r][idx] += 1 / len(refs)
            else:
                for r in refs:
                    dict_reference_counts[r][idx] += 1

    dict_references['num'] = dict_references['num'].astype(int)
    return (dict_references, dict_reference_counts)

################################################################################
### EMOTICONS

# TODO: This
class TweetEmoticons():
    def __init__(self,
            relative = True):
        self.names = []
        self.relative = relative