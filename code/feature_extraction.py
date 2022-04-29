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
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy import sparse
import heapq
import re
import pronouncing as pnc
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# default values for parameters of vectorizers
DEFAULT_CLEAN_TWEETS = False
DEFAULT_MAX_FEATURES = None
DEFAULT_BINARY = True
DEFAULT_RELATIVE = False
DEFAULT_ALPHABETIC = True
DEFAULT_SIMPLIFY = True

# Common stop words
DEFAULT_STOP_WORDS = ['the', 'th', 'this', 'to', 'you', 'and', 'in', 'is', 'will',
                      'that', 'on', 'of', 'just', 'it', 'for', 'be', 'at', 'about', 'going',
                      'yours', 'youre', 'your', 'youve', 'im', 'yet', 'yes', 'theres']
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


def clean_tweet(tweet):
    """Cleans a given tweet/tweet list. Removes:
    - Links: `http[s]://t.co/...`
    - Non-alphabetic characters: `[^a-z]`
    - Repeated/extraneous Spaces

    Args:
        tweet (str|list[str]): The tweet

    Returns:
        str: The cleaned tweet
    """

    if type(tweet) is not str: # clean the list of tweets
        return [clean_tweet(t) for t in tweet]

    new_tweet = tweet
    # replace urls with spaces
    new_tweet = re.sub(RE_LINKS, ' ', new_tweet)
    # replace non alphabetic characters with spaces
    new_tweet = re.sub(r'[^a-z]+', ' ', new_tweet)
    # reduce the spacing between words
    new_tweet = re.sub(r' +', ' ', new_tweet)
    # Remove bookend spaces
    new_tweet = re.sub(r'^ | $', '', new_tweet)
    return new_tweet


def generate_word_lists(tweets):
    """Split each tweet into a cleaned list of words

    Args:
        tweets (list[str]): The raw tweet list

    Returns:
        list[list[str]]: The list of lists of cleaned words
    """
    return [clean_tweet(t).split(' ') for t in tweets]


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

    def __init__(self, vectorizer_name='', names=[], clean_tweets=DEFAULT_CLEAN_TWEETS):
        """Create the TweetFeatureVectorizer

        Args:
            names (list, optional): Names of the features. Defaults to [].
            clean_tweets (bool, optional): Whether to clean the tweets before transforming. Defaults to DEFAULT_CLEAN_TWEETS.
        """
        self.vectorizer_name = vectorizer_name
        self.names = names
        self.clean_tweets = clean_tweets

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
        if self.clean_tweets:  # perform cleaning if necessary
            tweets = clean_tweet(tweets)
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
            stop_words=DEFAULT_STOP_WORDS,
            binary=DEFAULT_BINARY):

        super().__init__(vectorizer_name='bow')
        self.names = []
        self.params = {
            'max_features': max_features,
            'stop_words': stop_words,
            'binary': binary
        }
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words=stop_words,
            binary=binary)

    def get_params(self, deep=True):
        return self.params

    def fit(self, tweets, sentiments=None, **kwargs): return self

    def fit_transform(self, tweets, sentiments=None, **kwargs):
        out = self.vectorizer.fit_transform(clean_tweet(tweets))
        self.names = self.vectorizer.get_feature_names_out()
        return out

    def transform(self, tweets, sentiments=None, **kwargs):
        return self.vectorizer.transform(clean_tweet(tweets))

    def generate_dict(self, tweets, raw=False):
        t_matrix = None
        if raw:
            t_matrix = self.vectorizer.transform(clean_tweet(tweets))
        else:
            t_matrix = self.vectorizer.fit_transform(clean_tweet(tweets))

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
            stop_words=DEFAULT_STOP_WORDS):

        super().__init__(vectorizer_name='tfidf')
        self.names = []
        self.params = {
            'max_features': max_features,
            'stop_words': stop_words
        }
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=stop_words)

    def get_params(self, deep=True):
        return self.params

    def fit(self, tweets, sentiments=None, **kwargs): return self

    def fit_transform(self, tweets, sentiments=None, **kwargs):
        out = self.vectorizer.fit_transform(clean_tweet(tweets))
        self.names = self.vectorizer.get_feature_names_out()
        return out

    def transform(self, tweets, sentiments=None, **kwargs):
        return self.vectorizer.transform(clean_tweet(tweets))

    def generate_dict(self, tweets, raw=False):
        t_matrix = None
        if raw:
            t_matrix = self.vectorizer.transform(clean_tweet(tweets))
        else:
            t_matrix = self.vectorizer.fit_transform(clean_tweet(tweets))

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
    def __init__(self): super().__init__(vectorizer_name='metrics')

    # Create the dictionary of different length metrics
    def generate_dict(self, tweets, raw=False):
        # extract tweets and words
        word_lists = generate_word_lists(tweets)

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
        return out_dict


################################################################################
################################################################################
################################################################################
# WORD LENGTH DISTRIBUTIONS
class TweetWordLengths(TweetFeatureVectorizer):
    def __init__(
            self,
            max_features=DEFAULT_MAX_FEATURES,
            relative=DEFAULT_RELATIVE):

        super().__init__(vectorizer_name='word_lens', clean_tweets=True)
        self.names = []
        self.params = {
            'max_features': max_features,
            'relative': relative
        }

    def extract_features(self, tweet):
        return [len(w) for w in tweet.split(' ')]

################################################################################
################################################################################
################################################################################
# CHARACTER FREQUENCIES
class TweetCharacterFrequencies(TweetFeatureVectorizer):

    def __init__(
            self,
            max_features=DEFAULT_MAX_FEATURES,
            relative=DEFAULT_RELATIVE,
            alphabetic=DEFAULT_ALPHABETIC):

        super().__init__(vectorizer_name='char_freqs')
        self.names = []
        self.params = {
            'max_features': max_features,
            'relative': relative,
            'alphabetic': alphabetic
        }

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
            relative=DEFAULT_RELATIVE):

        super().__init__(vectorizer_name='links')
        self.names = []
        self.params = {
            'max_features': max_features,
            'relative': relative
        }

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
            relative=DEFAULT_RELATIVE):

        super().__init__(vectorizer_name='hashtag')
        self.names = []
        self.params = {
            'max_features': max_features,
            'relative': relative
        }

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
            relative=DEFAULT_RELATIVE):

        super().__init__(vectorizer_name='references')
        self.names = []
        self.params = {
            'max_features': max_features,
            'relative': relative
        }

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
            simplify=DEFAULT_SIMPLIFY):

        super().__init__(vectorizer_name='emotes')
        self.params = {
            'max_features': max_features,
            'relative': relative,
            'simplify': simplify
        }

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
            relative=DEFAULT_RELATIVE):

        super().__init__(vectorizer_name='phonetics', clean_tweets=True)
        self.names = []
        self.params = {
            'max_features': max_features,
            'relative': relative
        }

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

    def __init__(self): super().__init__(vectorizer_name='poetics')

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
