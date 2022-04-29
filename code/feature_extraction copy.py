""" This document contains all the functions necessary for the vectorization
of the tweet data. 
Currently implemented vectorizers are:
- An extensible TweetVectorizer superclass
- Bag-of-Words
- TF-IDF
- General count metrics (# words, # chars, # links, ...)
- Word length distributions
- Character Frequencies
- Link Frequencies
- Hashtag Frequencies
- Reference Frequencies
- Emoticons Frequencies (with arbitrary simplification)
- Phonetics Frequencies (using `pronouncing`)
- Poetic Phonetic Frequencies (using `pronouncing`)
"""
from pickle import STOP
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy import sparse
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import heapq
import re
import pronouncing as pnc
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# default values for parameters of vectorizers
DEFAULT_CLEAN_TWEETS = False
DEFAULT_REMOVE_REPEATED_CHARS = False
DEFAULT_STOP_WORDS = None
DEFAULT_SPLITTING_TYPE = None
DEFAULT_MAX_FEATURES = None
DEFAULT_BINARY = True
DEFAULT_RELATIVE = False
DEFAULT_ALPHABETIC = True
DEFAULT_SIMPLIFY = True

# Common stop words
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
STOP_WORDS = [
    'the', 'th', 'this', 'to', 'you', 'and', 'in', 'is', 'will', 'that', 'on', 
    'of', 'just', 'it', 'for', 'be', 'at', 'about', 'going', 'yours', 'youre', 
    'your', 'youve', 'im', 'yet', 'yes', 'theres', 'about', 'he', 'her', 'she', 
    'him', 'as', 'out', 'didn', 'who', 'whom', 'we', 'we', 're', 'see', 'st', 
    'll', 'time', 'all', 'my', 'if'
    ] + [c for c in ALPHABET]
# DEFAULT_STOP_WORDS = 'english'

# special term regex
RE_LINKS = r"https?://t.co/\w*"
RE_HASHTAGS = r"#\w+"
RE_REFERENCES = r"@\w+"

# define the sets of symbols to construct the emoticons
RE_EMO_EYES = r";:8="  # eye symbols
RE_EMO_MIDDLE = r"\',\-\"\*"  # middle symbols
RE_EMO_MOUTHS_HAP = r"p)3\]"  # happy/cutesy mouths
# if the emote is reversed these are happy mouths
RE_EMO_MOUTHS_HAP_BACK = r"(\["
RE_EMO_MOUTHS_SAD = r"\\/(\["  # sad/unhappy mouths
RE_EMO_MOUTHS_SAD_BACK = r"p)\]"  # if the emote is reversed these are sad mouths
RE_EMO_MOUTHS_SUR = r"vo"  # surprised mouths
RE_EMO_MOUTHS_NEU = r"l\|"  # neutral mouths
RE_EMO_MOUTHS = RE_EMO_MOUTHS_HAP + RE_EMO_MOUTHS_SAD + \
    RE_EMO_MOUTHS_SUR + RE_EMO_MOUTHS_NEU
# Only allow one type of mouth to be found at a time (`:\3` is not allowed)
RE_EMOTES = r"(?<=[ ^])[" + RE_EMO_EYES + r"]+[" + \
    RE_EMO_MIDDLE + r"]*[" + RE_EMO_MOUTHS_HAP + r"]+(?=[\W])|"
RE_EMOTES += r"(?<=[ ^])[" + RE_EMO_EYES + r"]+[" + \
    RE_EMO_MIDDLE + r"]*[" + RE_EMO_MOUTHS_SAD + r"]+(?=[\W])|"
RE_EMOTES += r"(?<=[ ^])[" + RE_EMO_EYES + r"]+[" + \
    RE_EMO_MIDDLE + r"]*[" + RE_EMO_MOUTHS_SUR + r"]+(?=[\W])|"
RE_EMOTES += r"(?<=[ ^])[" + RE_EMO_EYES + r"]+[" + \
    RE_EMO_MIDDLE + r"]*[" + RE_EMO_MOUTHS_NEU + r"]+(?=[\W])|"
# add the backwards results
RE_EMOTES += r"(?<=[ ^])[" + RE_EMO_MOUTHS_HAP_BACK + r"]+[" + \
    RE_EMO_MIDDLE + r"]*[" + RE_EMO_EYES + r"]+(?=[\W])|"
RE_EMOTES += r"(?<=[ ^])[" + RE_EMO_MOUTHS_SAD_BACK + r"]+[" + \
    RE_EMO_MIDDLE + r"]*[" + RE_EMO_EYES + r"]+(?=[\W])|"
RE_EMOTES += r"(?<=[ ^])[" + RE_EMO_MOUTHS_NEU + r"]+[" + \
    RE_EMO_MIDDLE + r"]*[" + RE_EMO_EYES + r"]+(?=[\W])"


def zero_default_dict(tweets):
    """Generates a default dict of zero arrays

    Args:
        tweets (list[str]): The raw tweet list
    """
    def zero_arr(): return np.zeros((len(tweets)))
    return defaultdict(zero_arr)


def clean_tweet(tweet, remove_repeated_chars=False, stop_words=None):
    """Cleans a given tweet/tweet list. Removes:
    - Links: `http[s]://t.co/...`
    - Non-alphabetic characters: `[^a-z]`
    - Repeated/extraneous Spaces

    Args:
        tweet (str|list[str]): The tweet or list of tweets
        remove_repeated_chars (bool, optional): Whether to remove duplicate consecutive characters. Defaults to `False`.

    Returns:
        str: The cleaned tweet
    """

    if type(tweet) is not str:  # clean the list of tweets
        return [clean_tweet(t, remove_repeated_chars, stop_words) for t in tweet]

    new_tweet = tweet
    # replace urls with spaces
    new_tweet = re.sub(RE_LINKS, ' ', new_tweet)
    # replace non alphabetic characters with spaces
    new_tweet = re.sub(r'[^a-z]+', ' ', new_tweet)
    # reduce the spacing between words
    new_tweet = re.sub(r' +', ' ', new_tweet)
    # remove stop_words
    if stop_words is not None:
        re_stop_words = r"(?<=\W)" + r"(?=\W)|(?<=\W)".join(stop_words) + r"(?=\W)"
        new_tweet = re.sub(re_stop_words, " ", new_tweet)
    # reduce repeated characters to one character
    if remove_repeated_chars:
        new_tweet = re.sub(r"(.)\1+", r"\1", new_tweet)
    # Remove bookend spaces
    new_tweet = re.sub(r'^ | $', '', new_tweet)
    return new_tweet

def split_tweet(tweet, splitter_type):
    if splitter_type is 'lemmatizer':
        wnl = WordNetLemmatizer()
        return [wnl.lemmatize(w) for w in tweet.split(' ')]
    elif splitter_type is 'stemmer':
        ps = PorterStemmer()
        return [ps.stem(w) for w in tweet.split('')]
    else:
        return tweet.split(' ')

def generate_word_lists(tweets, remove_repeated_chars=DEFAULT_REMOVE_REPEATED_CHARS, stop_words=DEFAULT_STOP_WORDS):  
    """Split each tweet into a cleaned list of words

    Args:
        tweets (list[str]): The raw tweet list
        remove_repeated_chars (bool, optional): Whether to remove duplicate consecutive characters. Defaults to `False`.

    Returns:
        list[list[str]]: The list of lists of cleaned words
    """
    return [clean_tweet(t, remove_repeated_chars, stop_words).spFlit(' ') for t in tweets]


def invert_feature_dict(values_dict, prefix=''):
    out_list = []
    out_keys = list(values_dict.keys())
    out_len = len(list(values_dict.values())[0])

    for idx in range(out_len):
        vector_dict = {}
        for key in out_keys:
            vector_dict[f'{prefix}.{key}'] = values_dict[key][idx]
        out_list.append(vector_dict)

    return out_list


# Find the max_features keys in a values dictionary (by the sum over their column)
def max_features_dict(values_dict, max_features=DEFAULT_MAX_FEATURES):
    """Find the `max_features` keys in a values dictionary (by the sum over their columns)

    Args:
        values_dict (dict): `(feature, array)` dictionary
        max_features (int, optional): Number of features to extract. Defaults to `DEFAULT_MAX_FEATURES`.

    Returns:
        dict: reduced `(feature, array)` dictionary
    """
    # manage scenarios where nothing needs to be done
    if max_features is None:
        return values_dict
    if max_features >= len(values_dict.keys()):
        return values_dict

    # develop a list of (value, key) tuples where value = sum(dict[key])
    top_list = []
    for (key, arr) in values_dict.items():
        heapq.heappush(top_list, (-1 * np.sum(arr), key))

    # now recreate the dictionary with the smaller list of features
    out_dict = dict()
    for r, name in top_list[:max_features]:
        out_dict[name] = values_dict[name]

    return out_dict

################################################################################
################################################################################
################################################################################
class TweetFeatureVectorizer(BaseEstimator):
    """A container/superclass for TweetFeatureVectorizers
        Most features (except for `Bag-of-Words` and `TF-IDF`) 
        use the same functions for `transform` and `fit_transform` and require a `generate_dict`

        Inherits from the `BaseEstimator` class
    """

    def __init__(self, vectorizer_name='',
                 clean_tweets=DEFAULT_CLEAN_TWEETS,
                 remove_repeated_chars=DEFAULT_REMOVE_REPEATED_CHARS,
                 stop_words=DEFAULT_STOP_WORDS):
        """Create the TweetFeatureVectorizer

        Args:
            vectorizer_name (str, optional): Name of the vectorizer. Defaults to ''.
            names (list, optional): The feature names. Defaults to [].
            clean_tweets (_type_, optional): Whether to clean tweets. Defaults to DEFAULT_CLEAN_TWEETS.
            remove_repeated_chars (_type_, optional): Whether to remove repeated characters. Defaults to DEFAULT_REMOVE_REPEATED_CHARS.
        """
        self.vectorizer_name = vectorizer_name
        self.names = []
        self.params = {}
        self.params['clean_tweets'] = clean_tweets
        self.params['remove_repeated_chars'] = remove_repeated_chars
        self.params['stop_words'] = stop_words

    def get_params(self, deep=True):
        return self.params

    def fit(self, tweets, sentiments=None, **kwargs): return self

    def fit_transform(self, tweets, sentiments=None, **kwargs):
        out_dict = self.generate_dict(tweets)
        self.names = list(out_dict.keys())
        # combine the arrays from the dictionary into the transformed matrix
        return np.matrix([arr for arr in out_dict.values()]).T

    def transform(self, tweets, sentiments=None, **kwargs):
        out_dict = self.generate_dict(tweets, raw=True)

        out_arrays = [out_dict[name] for name in self.names]

        # combine the arrays from the dictionary into the transformed matrix
        return np.matrix([arr for arr in out_arrays]).T

    def extract_features(self, tweet): return []

    def generate_dict(self, tweets, raw=False):
        if self.params['clean_tweets']:  # perform cleaning if necessary
            tweets = clean_tweet(tweets, self.params['remove_repeated_chars'], self.params['stop_words'])
        # create a default dict for occurrences of the feature
        out_dict = zero_default_dict(tweets)

        for idx, t in enumerate(tweets):
            features = self.extract_features(t)

            # the value by which to increment occurrences
            count = 1
            # check if its relative to the number of links in the tweet
            if self.params['relative']:
                count /= len(features)

            # iterate through the links and add to their counts
            for f in features:
                # this doesn't need to count values not in self.names and raw=true
                if raw and f not in self.names:
                    continue
                out_dict[f][idx] += count

        return max_features_dict(out_dict, self.params['max_features'])


################################################################################
################################################################################
################################################################################
# BAG OF WORDS
class TweetBagOfWords(TweetFeatureVectorizer):
    def __init__(
            self,
            max_features=DEFAULT_MAX_FEATURES,
            binary=DEFAULT_BINARY,
            clean_tweets=True,
            remove_repeated_chars=True, stop_words=STOP_WORDS,):

        super().__init__(vectorizer_name='bow', clean_tweets=clean_tweets,
                         remove_repeated_chars=remove_repeated_chars, stop_words=stop_words)
        
        self.params['max_features'] = max_features
        self.params['binary'] = binary
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words=stop_words,
            binary=binary)

    def get_params(self, deep=True):
        return self.params

    def fit(self, tweets, sentiments=None, **kwargs): return self

    def fit_transform(self, tweets, sentiments=None, **kwargs):
        if self.params['clean_tweets']:
            tweets = clean_tweet(tweets, self.params['remove_repeated_chars'], self.params['stop_words'])
        out = self.vectorizer.fit_transform(tweets)
        self.names = self.vectorizer.get_feature_names_out()
        return out

    def transform(self, tweets, sentiments=None, **kwargs):
        if self.params['clean_tweets']:
            tweets = clean_tweet(tweets, self.params['remove_repeated_chars'], self.params['stop_words'])
        return self.vectorizer.transform(tweets)

    def generate_dict(self, tweets, raw=False):
        if self.params['clean_tweets']:
            tweets = clean_tweet(tweets, self.params['remove_repeated_chars'], self.params['stop_words'])
        t_matrix = None
        if raw:
            t_matrix = self.vectorizer.transform(tweets)
        else:
            t_matrix = self.vectorizer.fit_transform(tweets)

        # create the dictionary per feature
        out_dict = zero_default_dict(tweets)
        vectorizer_words = self.names
        for id1, row in enumerate(t_matrix):
            for id2, val in zip(row.indices, row.data):
                out_dict[vectorizer_words[id2]][id1] = val

        return out_dict


################################################################################
################################################################################
################################################################################
# TF-IDF
class TweetTFIDF(TweetFeatureVectorizer):
    def __init__(
            self,
            max_features=DEFAULT_MAX_FEATURES,
            clean_tweets=True,
            remove_repeated_chars=True, stop_words=STOP_WORDS):

        super().__init__(vectorizer_name='tfidf', clean_tweets=clean_tweets,
                         remove_repeated_chars=remove_repeated_chars, stop_words=stop_words)
        
        self.params['max_features'] = max_features
        self.params['stop_words'] = stop_words
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=stop_words)

    def get_params(self, deep=True):
        return self.params

    def fit(self, tweets, sentiments=None, **kwargs): return self

    def fit_transform(self, tweets, sentiments=None, **kwargs):
        if self.params['clean_tweets']:
            tweets = clean_tweet(tweets, self.params['remove_repeated_chars'], self.params['stop_words'])
        out = self.vectorizer.fit_transform(tweets)
        self.names = self.vectorizer.get_feature_names_out()
        return out

    def transform(self, tweets, sentiments=None, **kwargs):
        if self.params['clean_tweets']:
            tweets = clean_tweet(tweets, self.params['remove_repeated_chars'], self.params['stop_words'])
        return self.vectorizer.transform(tweets)

    def generate_dict(self, tweets, raw=False):
        if self.params['clean_tweets']:
            tweets = clean_tweet(tweets, self.params['remove_repeated_chars'], self.params['stop_words'])
        t_matrix = None
        if raw:
            t_matrix = self.vectorizer.transform(tweets)
        else:
            t_matrix = self.vectorizer.fit_transform(tweets)

        # create the dictionary per feature
        out_dict = zero_default_dict(tweets)
        vectorizer_words = self.names
        for id1, row in enumerate(t_matrix):
            for id2, val in zip(row.indices, row.data):
                out_dict[vectorizer_words[id2]][id1] = val

        return out_dict

################################################################################
################################################################################
################################################################################
# NUMERIC METRICS
class TweetMetrics(TweetFeatureVectorizer):
    def __init__(self, clean_tweets=True, remove_repeated_chars=DEFAULT_REMOVE_REPEATED_CHARS,stop_words=DEFAULT_STOP_WORDS):
        super().__init__(vectorizer_name='metrics', clean_tweets=clean_tweets,
                         remove_repeated_chars=remove_repeated_chars, stop_words=stop_words)

    # Create the dictionary of different length metrics
    def generate_dict(self, tweets, raw=False):
        # extract tweets and words
        word_lists = generate_word_lists(tweets, self.params['remove_repeated_chars'])

        # create a default dict
        out_dict = zero_default_dict(tweets)

        for idx, (tweet, words) in enumerate(zip(tweets, word_lists)):
            out_dict['word'][idx] = len(words)
            out_dict['char'][idx] = len(tweet)
            tweet_alphanum = re.sub(r"[\W_]+", "", tweet)
            out_dict['alnum'][idx] = len(tweet_alphanum)
            tweet_alpha = re.sub(r"\d+", "", tweet_alphanum)
            out_dict['alpha'][idx] = len(tweet_alpha)
            out_dict['links'][idx] = len(
                re.findall(RE_LINKS, tweet))
            out_dict['hashtags'][idx] = len(
                re.findall(RE_HASHTAGS, tweet))
            out_dict['references'][idx] = len(
                re.findall(RE_REFERENCES, tweet))
            out_dict['emotes'][idx] = len(
                re.findall(RE_EMOTES, tweet))
            w_lengths = [len(w) for w in words]
            out_dict['average_word_length'][idx] = np.ma.average(w_lengths)
            out_dict['is_quoting'][idx] = ('"' in tweet)
        return out_dict


################################################################################
################################################################################
################################################################################
# WORD LENGTH DISTRIBUTIONS
class TweetWordLengths(TweetFeatureVectorizer):
    def __init__(
            self,
            max_features=DEFAULT_MAX_FEATURES,
            relative=DEFAULT_RELATIVE,
            clean_tweets=True,
            remove_repeated_chars=DEFAULT_REMOVE_REPEATED_CHARS,stop_words=DEFAULT_STOP_WORDS):

        super().__init__(vectorizer_name='word_lens', clean_tweets=clean_tweets,
                         remove_repeated_chars=remove_repeated_chars, stop_words=stop_words)
        
        self.params['max_features'] = max_features
        self.params['relative'] = relative

    def extract_features(self, tweet):
        return [len(w) for w in tweet.split(' ')]


################################################################################
################################################################################
################################################################################
# N-LENGTH GROUP WORDS
class TweetNWordGroups(TweetFeatureVectorizer):
    def __init__(
            self,
            max_features=DEFAULT_MAX_FEATURES,
            relative=DEFAULT_RELATIVE,
            clean_tweets=True,
            remove_repeated_chars=DEFAULT_REMOVE_REPEATED_CHARS, 
            stop_words=DEFAULT_STOP_WORDS,
            n = 2):

        super().__init__(vectorizer_name='word_lens', clean_tweets=clean_tweets,
                         remove_repeated_chars=remove_repeated_chars, stop_words=stop_words)
        
        self.params['max_features'] = max_features
        self.params['relative'] = relative
        self.params['n'] = n

    def extract_features(self, tweet):
        n = self.params['n']
        out = []
        words = tweet.split(' ')
        for idx in range(len(words)):
            if (len(words) - idx) > n: 
                n_group = ' '.join(words[idx:idx+n])
                out.append(n_group)
        return out


################################################################################
################################################################################
################################################################################
# CHARACTER FREQUENCIES
class TweetCharacterFrequencies(TweetFeatureVectorizer):

    def __init__(
            self,
            max_features=DEFAULT_MAX_FEATURES,
            relative=DEFAULT_RELATIVE,
            alphabetic=DEFAULT_ALPHABETIC,
            clean_tweets=DEFAULT_CLEAN_TWEETS,
            remove_repeated_chars=False, stop_wards=DEFAULT_STOP_WORDS):

        super().__init__(vectorizer_name='char_freqs', clean_tweets=clean_tweets,
                         remove_repeated_chars=remove_repeated_chars, stop_words=stop_words)
        
        self.params['max_features'] = max_features,
        self.params['relative'] = relative,
        self.params['alphabetic'] = alphabetic

    def extract_features(self, tweet):
        if self.params['alphabetic']:
            return re.sub(r"[^a-z]+", "", tweet)
        return tweet


################################################################################
################################################################################
################################################################################
# LINKS
class TweetLinks(TweetFeatureVectorizer):
    def __init__(
            self,
            max_features=DEFAULT_MAX_FEATURES,
            relative=DEFAULT_RELATIVE,
            clean_tweets=DEFAULT_CLEAN_TWEETS,
            remove_repeated_chars=DEFAULT_REMOVE_REPEATED_CHARS,stop_words=DEFAULT_STOP_WORDS):

        super().__init__(vectorizer_name='links', clean_tweets=clean_tweets,
                         remove_repeated_chars=remove_repeated_chars, stop_words=stop_words)
        
        
        self.params['max_features'] = max_features
        self.params['relative'] = relative

    def extract_features(self, tweet):
        return re.findall(RE_LINKS, tweet)


################################################################################
################################################################################
################################################################################
# HASHTAGS
class TweetHashtags(TweetFeatureVectorizer):
    def __init__(
            self,
            max_features=DEFAULT_MAX_FEATURES,
            relative=DEFAULT_RELATIVE,
            clean_tweets=DEFAULT_CLEAN_TWEETS,
            remove_repeated_chars=DEFAULT_REMOVE_REPEATED_CHARS,stop_words=DEFAULT_STOP_WORDS):

        super().__init__(vectorizer_name='hashtag', clean_tweets=clean_tweets,
                         remove_repeated_chars=remove_repeated_chars, stop_words=stop_words)
        
        self.params['max_features'] = max_features
        self.params['relative'] = relative

    def extract_features(self, tweet):
        return re.findall(RE_HASHTAGS, tweet)


################################################################################
################################################################################
################################################################################
# REFERENCES
class TweetReferences(TweetFeatureVectorizer):
    def __init__(
            self,
            max_features=DEFAULT_MAX_FEATURES,
            relative=DEFAULT_RELATIVE,
            clean_tweets=DEFAULT_CLEAN_TWEETS,
            remove_repeated_chars=DEFAULT_REMOVE_REPEATED_CHARS,stop_words=DEFAULT_STOP_WORDS):

        super().__init__(vectorizer_name='references', clean_tweets=clean_tweets,
                         remove_repeated_chars=remove_repeated_chars, stop_words=stop_words)
        
        self.params['max_features'] = max_features
        self.params['relative'] = relative

    def extract_features(self, tweet):
        return re.findall(RE_REFERENCES, tweet)


################################################################################
################################################################################
################################################################################
# EMOTICONS
class TweetEmoticons(TweetFeatureVectorizer):
    def __init__(
            self,
            max_features=DEFAULT_MAX_FEATURES,
            relative=DEFAULT_RELATIVE,
            simplify=DEFAULT_SIMPLIFY,
            clean_tweets=DEFAULT_CLEAN_TWEETS,
            remove_repeated_chars=DEFAULT_REMOVE_REPEATED_CHARS,stop_words=DEFAULT_STOP_WORDS):

        super().__init__(vectorizer_name='emotes', clean_tweets=clean_tweets,
                         remove_repeated_chars=remove_repeated_chars, stop_words=stop_words)
        self.params['max_features'] = max_features
        self.params['relative'] = relative
        self.params['simplify'] = simplify

    def extract_features(self, tweet):

        # extract valid emotes
        emotes = re.findall(RE_EMOTES, tweet)

        def isvalid(e):  # remove purely alphanumeric emotes
            return not (e.isalnum() or e.isdecimal() or e.isdigit())
        emotes = list(filter(isvalid, emotes))

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
                    if symbol == e_simple[-1]:
                        continue
                    if symbol in RE_EMO_MOUTHS and e_simple[-1] in RE_EMO_MOUTHS:
                        continue
                e_simple += symbol

            return e_simple

        # simplify the emotes as needed
        if self.params['simplify']:
            emotes = [simplify_emoticon(e) for e in emotes]

        return emotes


################################################################################
################################################################################
################################################################################
# PHONETICS
class TweetPhonetics(TweetFeatureVectorizer):
    def __init__(
            self,
            max_features=DEFAULT_MAX_FEATURES,
            relative=DEFAULT_RELATIVE,
            clean_tweets=True,
            remove_repeated_chars=False, stop_words=None):

        super().__init__(vectorizer_name='phonetics', clean_tweets=clean_tweets,
                         remove_repeated_chars=remove_repeated_chars, stop_words=stop_words)
        
        self.params['max_features'] = max_features
        self.params['relative'] = relative

    def extract_features(self, tweet):

        words = tweet.split(' ')
        total_phones = []

        for w in words:
            # Break the word into its phones
            phones = pnc.phones_for_word(w)

            # break the phones into a list and add to it
            if len(phones) > 0:
                total_phones += phones[0].split(' ')

        return total_phones


################################################################################
################################################################################
################################################################################
# POETIC PHONETICS
class TweetPoetics(TweetFeatureVectorizer):

    def __init__(self,
                 clean_tweets=True,
                 remove_repeated_chars=False, stop_words=None):
        super().__init__(vectorizer_name='poetics')

    # create a breakdown of the phonetics used in the tweets
    def generate_dict(self, tweets, raw=False):
        # create the phonetics dictionary
        out_dict = zero_default_dict(tweets)

        # sets of poetic noises
        # reference: https://nlp.stanford.edu/courses/lsa352/arpabet.html
        plosive_set = ['B', 'P', 'D', 'T', 'G', 'K',
                       'BCL', 'PCL', 'DCL', 'TCL', 'GCL', 'KCL', 'DX']
        re_plosive = r"(?<=[ ^])" + \
            r"(?=[ $])|(?<=[ ^])".join(plosive_set) + r"(?=[ $])"
        fricative_hard_set = ['DH', 'V']
        re_fricative_hard = r"(?<=[ ^])" + \
            r"(?=[ $])|(?<=[ ^])".join(fricative_hard_set) + r"(?=[ $])"
        fricative_soft_set = ['TH', 'F']
        re_fricative_soft = r"(?<=[ ^])" + \
            r"(?=[ $])|(?<=[ ^])".join(fricative_soft_set) + r"(?=[ $])"
        sibilant_set = ['Z', 'S', 'CH', 'SH', 'ZH']
        re_sibilant = r"(?<=[ ^])" + \
            r"(?=[ $])|(?<=[ ^])".join(sibilant_set) + r"(?=[ $])"

        # break the tweets into word lists
        word_lists = generate_word_lists(tweets)

        # iterate through tweets and extract the emoticons
        for idx, words in enumerate(word_lists):
            # iterate through the words
            for w in words:
                # Break the word into its phones
                phones = pnc.phones_for_word(w)
                if len(phones) > 0:
                    phones = phones[0]
                else:
                    continue  # skip words with no phones

                if re.search(re_plosive, phones) is not None:
                    out_dict["plosive"][idx] += 1

                if re.search(re_sibilant, phones) is not None:
                    out_dict["sibilant"][idx] += 1

                if re.search(re_fricative_hard, phones) is not None:
                    out_dict["fricative"][idx] += 1
                    out_dict["fricative-hard"][idx] += 1

                if re.search(re_fricative_soft, phones) is not None:
                    out_dict["fricative"][idx] += 1
                    out_dict["fricative-soft"][idx] += 1

        return out_dict
