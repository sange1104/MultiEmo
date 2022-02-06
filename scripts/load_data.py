''' 
Implements the tokenizing and additional preprocessing from torchmoji repo
https://github.com/huggingface/torchMoji/tree/master/torchmoji
'''

from copy import deepcopy
import pandas as pd
import numpy as np
import json
import re
import string
from text_unidecode import unidecode
import unicodedata
from torch.utils.data import Dataset, DataLoader
import torch


VOCAB_PATH = './data/vocabulary.json'
with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)

ALLOWED_CONVERTED_UNICODE_PUNCTUATION = """!"#$'()+,-.:;<=>?@`~"""

# Basic patterns.
RE_NUM = r'[0-9]+'
RE_WORD = r'[a-zA-Z]+'
RE_WHITESPACE = r'\s+'
RE_ANY = r'.'

# Combined words such as 'red-haired' or 'CUSTOM_TOKEN'
RE_COMB = r'[a-zA-Z]+[-_][a-zA-Z]+'

# English-specific patterns
RE_CONTRACTIONS = RE_WORD + r'\'' + RE_WORD

TITLES = [
    r'Mr\.',
    r'Ms\.',
    r'Mrs\.',
    r'Dr\.',
    r'Prof\.',
    ]
# Ensure case insensitivity
RE_TITLES = r'|'.join([r'(?i)' + t for t in TITLES])

# Symbols have to be created as separate patterns in order to match consecutive
# identical symbols.
SYMBOLS = r'()<!?.,/\'\"-_=\\§|´ˇ°[]<>{}~$^&*;:%+\xa3€`'
RE_SYMBOL = r'|'.join([re.escape(s) + r'+' for s in SYMBOLS])

# Hash symbols and at symbols have to be defined separately in order to not
# clash with hashtags and mentions if there are multiple - i.e.
# ##hello -> ['#', '#hello'] instead of ['##', 'hello']
SPECIAL_SYMBOLS = r'|#+(?=#[a-zA-Z0-9_]+)|@+(?=@[a-zA-Z0-9_]+)|#+|@+'
RE_SYMBOL += SPECIAL_SYMBOLS

RE_ABBREVIATIONS = r'\b(?<!\.)(?:[A-Za-z]\.){2,}'

# Twitter-specific patterns
RE_HASHTAG = r'#[a-zA-Z0-9_]+'
RE_MENTION = r'@[a-zA-Z0-9_]+'

RE_URL = r'(?:https?://|www\.)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
RE_EMAIL = r'\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b'

unicode = str
SPECIAL_TOKENS = ['CUSTOM_MASK',
                  'CUSTOM_UNKNOWN',
                  'CUSTOM_AT',
                  'CUSTOM_URL',
                  'CUSTOM_NUMBER',
                  'CUSTOM_BREAK']

# from http://bit.ly/2rdjgjE (UTF-8 encodings and Unicode chars)
VARIATION_SELECTORS = [ '\ufe00',
                        '\ufe01',
                        '\ufe02',
                        '\ufe03',
                        '\ufe04',
                        '\ufe05',
                        '\ufe06',
                        '\ufe07',
                        '\ufe08',
                        '\ufe09',
                        '\ufe0a',
                        '\ufe0b',
                        '\ufe0c',
                        '\ufe0d',
                        '\ufe0e',
                        '\ufe0f']

# Emoticons and emojis
RE_HEART = r'(?:<+/?3+)+'
EMOTICONS_START = [
    r'>:',
    r':',
    r'=',
    r';',
    ]
EMOTICONS_MID = [
    r'-',
    r',',
    r'^',
    '\'',
    '\"',
    ]
EMOTICONS_END = [
    r'D',
    r'd',
    r'p',
    r'P',
    r'v',
    r')',
    r'o',
    r'O',
    r'(',
    r'3',
    r'/',
    r'|',
    '\\',
    ]
EMOTICONS_EXTRA = [
    r'-_-',
    r'x_x',
    r'^_^',
    r'o.o',
    r'o_o',
    r'(:',
    r'):',
    r');',
    r'(;',
    ]

RE_EMOTICON = r'|'.join([re.escape(s) for s in EMOTICONS_EXTRA])
for s in EMOTICONS_START:
    for m in EMOTICONS_MID:
        for e in EMOTICONS_END:
            RE_EMOTICON += '|{0}{1}?{2}+'.format(re.escape(s), re.escape(m), re.escape(e))

# requires ucs4 in python2.7 or python3+
# RE_EMOJI = r"""[\U0001F300-\U0001F64F\U0001F680-\U0001F6FF\u2600-\u26FF\u2700-\u27BF]"""
# safe for all python
RE_EMOJI = r"""\ud83c[\udf00-\udfff]|\ud83d[\udc00-\ude4f\ude80-\udeff]|[\u2600-\u26FF\u2700-\u27BF]"""

# List of matched token patterns, ordered from most specific to least specific.
TOKENS = [
    RE_URL,
    RE_EMAIL,
    RE_COMB,
    RE_HASHTAG,
    RE_MENTION,
    RE_HEART,
    RE_EMOTICON,
    RE_CONTRACTIONS,
    RE_TITLES,
    RE_ABBREVIATIONS,
    RE_NUM,
    RE_WORD,
    RE_SYMBOL,
    RE_EMOJI,
    RE_ANY
    ]

# List of ignored token patterns
IGNORED = [
    RE_WHITESPACE
    ]

RE_MENTION = r'@[a-zA-Z0-9_]+'
AtMentionRegex = re.compile(RE_MENTION)

RE_URL = r'(?:https?://|www\.)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
urlRegex = re.compile(RE_URL)

# Final pattern
RE_PATTERN = re.compile(r'|'.join(IGNORED) + r'|(' + r'|'.join(TOKENS) + r')',
                        re.UNICODE)

def tokenize(text):
    '''Splits given input string into a list of tokens.

    # Arguments:
        text: Input string to be tokenized.

    # Returns:
        List of strings (tokens).
    '''
    result = RE_PATTERN.findall(text)

    # Remove empty strings
    result = [t for t in result if t.strip()]
    return result

def convert_linebreaks(text):
    # ugly hack handling non-breaking space no matter how badly it's been encoded in the input
    # space around to ensure proper tokenization
    for r in ['\\\\n', '\\n', '\n', '\\\\r', '\\r', '\r', '<br>']:
        text = text.replace(r, ' ' + SPECIAL_TOKENS[5] + ' ')
    return text

def punct_word(word, punctuation=string.punctuation):
    return all([True if c in punctuation else False for c in word])

def remove_variation_selectors(text):
    """ Remove styling glyph variants for Unicode characters.
        For instance, remove skin color from emojis.
    """
    for var in VARIATION_SELECTORS:
        text = text.replace(var, '')
    return text

def process_word(word):
    """ Shortening and converting the word to a special token if relevant.
    """
    word = shorten_word(word)
    word = detect_special_tokens(word)
    return word


def detect_special_tokens(word):
    try:
        int(word)
        word = SPECIAL_TOKENS[4]
    except ValueError:
        if AtMentionRegex.findall(word):
            word = SPECIAL_TOKENS[2]
        elif urlRegex.findall(word):
            word = SPECIAL_TOKENS[3]
    return word 

def shorten_word(word):
    """ Shorten groupings of 3+ identical consecutive chars to 2, e.g. '!!!!' --> '!!'
    """

    # only shorten ASCII words
    try:
        word.decode('ascii')
    except (UnicodeDecodeError, UnicodeEncodeError, AttributeError) as e:
        return word

    # must have at least 3 char to be shortened
    if len(word) < 3:
        return word

    # find groups of 3+ consecutive letters
    letter_groups = [list(g) for k, g in groupby(word)]
    triple_or_more = [''.join(g) for g in letter_groups if len(g) >= 3]
    if len(triple_or_more) == 0:
        return word

    # replace letters to find the short word
    short_word = word
    for trip in triple_or_more:
        short_word = short_word.replace(trip, trip[0]*2)

    return short_word

class WordGenerator():
    ''' Cleanses input and converts into words. Needs all sentences to be in
        Unicode format. Has subclasses that read sentences differently based on
        file type.

    Takes a generator as input. This can be from e.g. a file.
    unicode_handling in ['ignore_sentence', 'convert_punctuation', 'allow']
    unicode_handling in ['ignore_emoji', 'ignore_sentence', 'allow']
    '''
    def __init__(self, stream, allow_unicode_text=False, ignore_emojis=True,
                 remove_variation_selectors=True, break_replacement=True):
        self.stream = stream
        self.allow_unicode_text = allow_unicode_text
        self.remove_variation_selectors = remove_variation_selectors
        self.ignore_emojis = ignore_emojis
        self.break_replacement = break_replacement
        self.reset_stats()

    def get_words(self, sentence):
        """ Tokenizes a sentence into individual words.
            Converts Unicode punctuation into ASCII if that option is set.
            Ignores sentences with Unicode if that option is set.
            Returns an empty list of words if the sentence has Unicode and
            that is not allowed.
        """

        if not isinstance(sentence, unicode):
            raise ValueError("All sentences should be Unicode-encoded!")
        sentence = sentence.strip().lower()

        if self.break_replacement:
            sentence = convert_linebreaks(sentence)

        if self.remove_variation_selectors:
            sentence = remove_variation_selectors(sentence)

        # Split into words using simple whitespace splitting and convert
        # Unicode. This is done to prevent word splitting issues with
        # twokenize and Unicode
        words = sentence.split()
        converted_words = []
        for w in words:
            accept_sentence, c_w = self.convert_unicode_word(w)
            # Unicode word detected and not allowed
            if not accept_sentence:
                return []
            else:
                converted_words.append(c_w)
        sentence = ' '.join(converted_words)

        words = tokenize(sentence)
        words = [process_word(w) for w in words]
        return words

    def check_ascii(self, word):
        """ Returns whether a word is ASCII """

        try:
            word.decode('ascii')
            return True
        except (UnicodeDecodeError, UnicodeEncodeError, AttributeError):
            return False

    def convert_unicode_punctuation(self, word):
        word_converted_punct = []
        for c in word:
            decoded_c = unidecode(c).lower()
            if len(decoded_c) == 0:
                # Cannot decode to anything reasonable
                word_converted_punct.append(c)
            else:
                # Check if all punctuation and therefore fine
                # to include unidecoded version
                allowed_punct = punct_word(
                        decoded_c,
                        punctuation=ALLOWED_CONVERTED_UNICODE_PUNCTUATION)

                if allowed_punct:
                    word_converted_punct.append(decoded_c)
                else:
                    word_converted_punct.append(c)
        return ''.join(word_converted_punct)

    def convert_unicode_word(self, word):
        """ Converts Unicode words to ASCII using unidecode. If Unicode is not
            allowed (set as a variable during initialization), then only
            punctuation that can be converted to ASCII will be allowed.
        """
        if self.check_ascii(word):
            return True, word

        # First we ensure that the Unicode is normalized so it's
        # always a single character.
        word = unicodedata.normalize("NFKC", word)

        # Convert Unicode punctuation to ASCII equivalent. We want
        # e.g. "\u203c" (double exclamation mark) to be treated the same
        # as "!!" no matter if we allow other Unicode characters or not.
        word = self.convert_unicode_punctuation(word)

        if self.ignore_emojis:
            _, word = separate_emojis_and_text(word)

        # If conversion of punctuation and removal of emojis took care
        # of all the Unicode or if we allow Unicode then everything is fine
        if self.check_ascii(word) or self.allow_unicode_text:
            return True, word
        else:
            # Sometimes we might want to simply ignore Unicode sentences
            # (e.g. for vocabulary creation). This is another way to prevent
            # "polution" of strange Unicode tokens from low quality datasets
            return False, ''

    def data_preprocess_filtering(self, line, iter_i):
        """ To be overridden with specific preprocessing/filtering behavior
            if desired.

            Returns a boolean of whether the line should be accepted and the
            preprocessed text.

            Runs prior to tokenization.
        """
        return True, line, {}

    def data_postprocess_filtering(self, words, iter_i):
        """ To be overridden with specific postprocessing/filtering behavior
            if desired.

            Returns a boolean of whether the line should be accepted and the
            postprocessed text.

            Runs after tokenization.
        """
        return True, words, {}

    def extract_valid_sentence_words(self, line):
        """ Line may either a string of a list of strings depending on how
            the stream is being parsed.
            Domain-specific processing and filtering can be done both prior to
            and after tokenization.
            Custom information about the line can be extracted during the
            processing phases and returned as a dict.
        """

        info = {}

        pre_valid, pre_line, pre_info = \
            self.data_preprocess_filtering(line, self.stats['total'])
        info.update(pre_info)
        if not pre_valid:
            self.stats['pretokenization_filtered'] += 1
            return False, [], info

        words = self.get_words(pre_line)
        if len(words) == 0:
            self.stats['unicode_filtered'] += 1
            return False, [], info

        post_valid, post_words, post_info = \
            self.data_postprocess_filtering(words, self.stats['total'])
        info.update(post_info)
        if not post_valid:
            self.stats['posttokenization_filtered'] += 1
        return post_valid, post_words, info

    def generate_array_from_input(self):
        sentences = []
        for words in self:
            sentences.append(words)
        return sentences

    def reset_stats(self):
        self.stats = {'pretokenization_filtered': 0,
                      'unicode_filtered': 0,
                      'posttokenization_filtered': 0,
                      'total': 0,
                      'valid': 0}

    def __iter__(self):
        if self.stream is None:
            raise ValueError("Stream should be set before iterating over it!")

        for line in self.stream:
            valid, words, info = self.extract_valid_sentence_words(line)

            # Words may be filtered away due to unidecode etc.
            # In that case the words should not be passed on.
            if valid and len(words):
                self.stats['valid'] += 1
                yield words, info

            self.stats['total'] += 1

def remove_neutral(df):
    '''GoEmotion - drop the data labeled as "neutral"''' 
    drop_idx = df[df.label_27==27].index
    df = df.drop(drop_idx)
    df = df.reset_index()
    df = df.drop('index', axis=1)
    return df

class InferDataset(Dataset):
    def __init__(self, text, tokenizer):  
        self.text = text    
        self.tokenizer = tokenizer 

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = self.text[item]  
        tokens = self.text[item].split(' ')  
         
        tokens, _, _ = self.tokenizer.tokenize_sentences([text]) 
        
        return { 
          'token': tokens[0].astype('int64') 
        }

class MultiDataset(Dataset):
    def __init__(self, text, label, max_len, tokenizer, aux_task):
        self.text = text
        self.label = label 
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.task = aux_task

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = self.text[item]
        label = self.label[item] 

        tokens = self.text[item].split(' ') 
        task = self.task[item]
         
        tokens, _, _ = self.tokenizer.tokenize_sentences([text]) 
        
        return {
          'label': torch.tensor(label, dtype=torch.int64),
          'token': tokens[0].astype('int64'),
          'task':torch.tensor(task, dtype=torch.int64)
        }

def create_data_loader(aux_num, aux_task, df_emoji, df_emotion, max_len, batch_size, tokenizer):
    task_to_idx = {'emo':1, 'Ekman':2, 'sent':3}
    if aux_num == 0:
        # single-task dataset
        pass
    else:
        # preprocess
        df_emotion = remove_neutral(df_emotion)

        # multi-task dataset
        task_to_labcol = {'emo' : df_emotion.label_27, 
                          'Ekman' : df_emotion.label_7, 
                          'sent' : df_emotion.label_4} 
          
        # initialize text, label, task list using for emoji prediction
        text = list(df_emoji.pre_punc)
        label = list(df_emoji.label_int)
        task = [0 for _ in range(len(df_emoji))]
        
        # construct additional set for auxiliary task
        for aux in aux_task: # ex. ['emo', 'sent']
            text += list(df_emotion.text)
            label += list(task_to_labcol[aux])
            task += [task_to_idx[aux] for _ in range(len(df_emotion))]
        
        # construct pre-defined dataset
        dataset = MultiDataset(text = np.array(text),
                              label = np.array(label), 
                              max_len = max_len,
                              tokenizer = tokenizer,
                              aux_task = np.array(task))

    return DataLoader(
        dataset,
        batch_size = batch_size, 
        drop_last = True,
        shuffle = True
        )

class SentenceTokenizer():
    """ Create numpy array of tokens corresponding to input sentences.
        The vocabulary can include Unicode tokens.
    """
    def __init__(self, fixed_length, custom_wordgen=None,
                 ignore_sentences_with_only_custom=False, masking_value=0,
                 unknown_value=1):
        """ Needs a dictionary as input for the vocabulary.
        """

        if len(vocabulary) > np.iinfo('uint16').max:
            raise ValueError('Dictionary is too big ({} tokens) for the numpy '
                             'datatypes used (max limit={}). Reduce vocabulary'
                             ' or adjust code accordingly!'
                             .format(len(vocabulary), np.iinfo('uint16').max))

        # Shouldn't be able to modify the given vocabulary
        self.vocabulary = deepcopy(vocabulary)
        self.fixed_length = fixed_length
        self.ignore_sentences_with_only_custom = ignore_sentences_with_only_custom
        self.masking_value = masking_value
        self.unknown_value = unknown_value

        # Initialized with an empty stream of sentences that must then be fed
        # to the generator at a later point for reusability.
        # A custom word generator can be used for domain-specific filtering etc
        if custom_wordgen is not None:
            assert custom_wordgen.stream is None
            self.wordgen = custom_wordgen
            self.uses_custom_wordgen = True
        else:
            self.wordgen = WordGenerator(None, allow_unicode_text=True,
                                         ignore_emojis=False,
                                         remove_variation_selectors=True,
                                         break_replacement=True)
            self.uses_custom_wordgen = False

    def tokenize_sentences(self, sentences, reset_stats=True, max_sentences=None):
        """ Converts a given list of sentences into a numpy array according to
            its vocabulary.

        # Arguments:
            sentences: List of sentences to be tokenized.
            reset_stats: Whether the word generator's stats should be reset.
            max_sentences: Maximum length of sentences. Must be set if the
                length cannot be inferred from the input.

        # Returns:
            Numpy array of the tokenization sentences with masking,
            infos,
            stats

        # Raises:
            ValueError: When maximum length is not set and cannot be inferred.
        """

        if max_sentences is None and not hasattr(sentences, '__len__'):
            raise ValueError('Either you must provide an array with a length'
                             'attribute (e.g. a list) or specify the maximum '
                             'length yourself using `max_sentences`!')
        n_sentences = (max_sentences if max_sentences is not None
                       else len(sentences))

        if self.masking_value == 0:
            tokens = np.zeros((n_sentences, self.fixed_length), dtype='uint16')
        else:
            tokens = (np.ones((n_sentences, self.fixed_length), dtype='uint16')
                      * self.masking_value)

        if reset_stats:
            self.wordgen.reset_stats()

        # With a custom word generator info can be extracted from each
        # sentence (e.g. labels)
        infos = []

        # Returns words as strings and then map them to vocabulary
        self.wordgen.stream = sentences
        next_insert = 0
        n_ignored_unknowns = 0
        for s_words, s_info in self.wordgen:
            s_tokens = self.find_tokens(s_words)

            if (self.ignore_sentences_with_only_custom and
                np.all([True if t < len(SPECIAL_TOKENS)
                        else False for t in s_tokens])):
                n_ignored_unknowns += 1
                continue
            if len(s_tokens) > self.fixed_length:
                s_tokens = s_tokens[:self.fixed_length]
            tokens[next_insert,:len(s_tokens)] = s_tokens
            infos.append(s_info)
            next_insert += 1

        # For standard word generators all sentences should be tokenized
        # this is not necessarily the case for custom wordgenerators as they
        # may filter the sentences etc.
        if not self.uses_custom_wordgen and not self.ignore_sentences_with_only_custom:
            assert len(sentences) == next_insert
        else:
            # adjust based on actual tokens received
            tokens = tokens[:next_insert]
            infos = infos[:next_insert]

        return tokens, infos, self.wordgen.stats

    def find_tokens(self, words):
        assert len(words) > 0
        tokens = []
        for w in words:
            try:
                tokens.append(self.vocabulary[w])
            except KeyError:
                tokens.append(self.unknown_value)
        return tokens

    def split_train_val_test(self, sentences, info_dicts,
                             split_parameter=[0.7, 0.1, 0.2], extend_with=0):
        """ Splits given sentences into three different datasets: training,
            validation and testing.

        # Arguments:
            sentences: The sentences to be tokenized.
            info_dicts: A list of dicts that contain information about each
                sentence (e.g. a label).
            split_parameter: A parameter for deciding the splits between the
                three different datasets. If instead of being passed three
                values, three lists are passed, then these will be used to
                specify which observation belong to which dataset.
            extend_with: An optional parameter. If > 0 then this is the number
                of tokens added to the vocabulary from this dataset. The
                expanded vocab will be generated using only the training set,
                but is applied to all three sets.

        # Returns:
            List of three lists of tokenized sentences,

            List of three corresponding dictionaries with information,

            How many tokens have been added to the vocab. Make sure to extend
            the embedding layer of the model accordingly.
        """

        # If passed three lists, use those directly
        if isinstance(split_parameter, list) and \
                all(isinstance(x, list) for x in split_parameter) and \
                len(split_parameter) == 3:

            # Helper function to verify provided indices are numbers in range
            def verify_indices(inds):
                return list(filter(lambda i: isinstance(i, numbers.Number)
                            and i < len(sentences), inds))

            ind_train = verify_indices(split_parameter[0])
            ind_val = verify_indices(split_parameter[1])
            ind_test = verify_indices(split_parameter[2])
        else:
            # Split sentences and dicts
            ind = list(range(len(sentences)))
            ind_train, ind_test = train_test_split(ind, test_size=split_parameter[2])
            ind_train, ind_val = train_test_split(ind_train, test_size=split_parameter[1])

        # Map indices to data
        train = np.array([sentences[x] for x in ind_train])
        test = np.array([sentences[x] for x in ind_test])
        val = np.array([sentences[x] for x in ind_val])

        info_train = np.array([info_dicts[x] for x in ind_train])
        info_test = np.array([info_dicts[x] for x in ind_test])
        info_val = np.array([info_dicts[x] for x in ind_val])

        added = 0
        # Extend vocabulary with training set tokens
        if extend_with > 0:
            wg = WordGenerator(train)
            vb = VocabBuilder(wg)
            vb.count_all_words()
            added = extend_vocab(self.vocabulary, vb, max_tokens=extend_with)

        # Wrap results
        result = [self.tokenize_sentences(s)[0] for s in [train, val, test]]
        result_infos = [info_train, info_val, info_test]
        # if type(result_infos[0][0]) in [np.double, np.float, np.int64, np.int32, np.uint8]:
        #     result_infos = [torch.from_numpy(label).long() for label in result_infos]

        return result, result_infos, added

    def to_sentence(self, sentence_idx):
        """ Converts a tokenized sentence back to a list of words.

        # Arguments:
            sentence_idx: List of numbers, representing a tokenized sentence
                given the current vocabulary.

        # Returns:
            String created by converting all numbers back to words and joined
            together with spaces.
        """
        # Have to recalculate the mappings in case the vocab was extended.
        ind_to_word = {ind: word for word, ind in self.vocabulary.items()}

        sentence_as_list = [ind_to_word[x] for x in sentence_idx]
        cleaned_list = [x for x in sentence_as_list if x != 'CUSTOM_MASK']
        return " ".join(cleaned_list)

