import pandas as pd
import numpy as np   
import numpy as np 
import string
import os
import re
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split 
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons 
import argparse

def load_data(path):
    df = pd.read_csv(path, header=None) 
    return df

def clean_data(df):
    df = df.drop_duplicates(subset=['original'])
    df = df.reset_index() 
    return df

def get_text(df):  
    text = df['original']
    return text

class TextProcessor:
    '''
    Implements TextProcessor from https://gist.github.com/luisfredgs/4ee62c6f3843cbdbf1790748b724716c
    '''
    def __init__(self):
        self.text_processor = self.get_text_processor()  
        self.emojis = '😌💯😤😀💓📷👑😇😱😭💔♻😩➡✔😔😃😳💗😊💛🌸😫✅👏🙊☺😎🙈😒💪😴👍😅😏⬅👈💚😝👀🎶✋😐😍😢😁💀😡💕👌😣💞💘♥👇🌹😑💋🙌😞▶❤💜😕🙏😈💁💖✌😄😬😉👉😆😪😜✨🌚🎉💙😂☀😘🔥😋'
        
    def get_text_processor(self):

        text_processor = TextPreProcessor(
            # terms that will be normalized
            normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                       'time', 'url', 'date', 'number'],
            # terms that will be annotated
            # annotate={"hashtag", "allcaps", "elongated", "repeated",
            #           'emphasis', 'censored'},

            # annotate={"repeated", "emphasis", "elongated"},

            annotate={"allcaps", "elongated", #  "repeated",
                          'emphasis', 'censored'},

            fix_html=True,  # fix HTML tokens

            # corpus from which the word statistics are going to be used
            # for word segmentation
            segmenter="twitter",

            # corpus from which the word statistics are going to be used
            # for spell correction
            corrector="twitter",

            unpack_hashtags=True,  # perform word segmentation on hashtags
            unpack_contractions=True,  # Unpack contractions (can't -> can not)
            spell_correct_elong=True,  # spell correction for elongated words

            # select a tokenizer. You can use SocialTokenizer, or pass your own
            # the tokenizer, should take as input a string and return a list of tokens
            tokenizer=SocialTokenizer(lowercase=True).tokenize,

            # list of dictionaries, for replacing tokens extracted from the text,
            # with other expressions. You can pass more than one dictionaries.
            dicts=[emoticons]
        )
        return text_processor
    
    def processing_pipeline(self, text):
        text = text.replace('\n', ' ') 
        text = text.replace('RT',' ')
        # stop_words = set(stopwords.words('english'))
        n=WordNetLemmatizer()
        text = ' '.join([n.lemmatize(word) for word in text.split() if (not(word.startswith('@')) and (not(word.startswith('#'))))])
        text = text.lower().strip() 
        text_split = ' '.join(self.text_processor.pre_process_doc(text))
        if '<url>' in text_split:
            return None
        text = re.sub("[^a-zA-Z?!.😌💯😤😀💓📷👑😇😱😭💔♻😩➡✔😔😃😳💗😊💛🌸😫✅👏🙊☺😎🙈😒💪😴👍😅😏⬅👈💚😝👀🎶✋😐😍😢😁💀😡💕👌😣💞💘♥👇🌹😑💋🙌😞▶❤💜😕🙏😈💁💖✌😄😬😉👉😆😪😜✨🌚🎉💙😂☀😘🔥😋]"," ", text_split)
        # text = ' '.join([word for word in text.split(' ') if (len(word)>1 or word in self.emojis)])
        # if len(text.split(' ')) < 5: # more than 5 tokens
        #     return None
        return text



def preprocess(text, preprocessor): 
    emojis = '😌💯😤😀💓📷👑😇😱😭💔♻😩➡✔😔😃😳💗😊💛🌸😫✅👏🙊☺😎🙈😒💪😴👍😅😏⬅👈💚😝👀🎶✋😐😍😢😁💀😡💕👌😣💞💘♥👇🌹😑💋🙌😞▶❤💜😕🙏😈💁💖✌😄😬😉👉😆😪😜✨🌚🎉💙😂☀😘🔥😋'   
    pre_text = preprocessor.processing_pipeline(text) 
    if pre_text!=None:
        emoji_label = []
        pre_text = pre_text.replace('!', ' !')
        pre_text = pre_text.replace('?', ' ?')
        pre_text = pre_text.replace('.', ' .')

        original = pre_text.split(' ')
        original.remove('')
        
        #split emoji from text
        for emoji in emojis:
            if emoji in pre_text:
                emoji_label.append(emoji)
                pre_text = pre_text.replace(emoji, '')
        original = [w for w in original if w!='']
        pre_text = [w for w in pre_text.split(' ') if w!='']
        if  len(pre_text)>1:
            return (' '.join(pre_text), ' '.join(original), ''.join(emoji_label))

def __main__(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--data_path', required=True) 
    args = parser.parse_args()

    text_path = args.data_path
    assert os.path.exists(text_path), 'Data does not exist in the path.' 
    
    emoji_to_idx = {'✅': 0, '✨': 1, '🌚': 2, '🎉': 3, '🎶': 4, '👀': 5, '👇': 6, '👌': 7, '👍': 8, '👏': 9, '👑': 10, '💀': 11, '💔': 12, '💕': 13, '💖': 14, '💗': 15, '💙': 16,\
                    '💚': 17, '💛': 18, '💜': 19, '💞': 20, '💪': 21, '💯': 22, '🔥': 23, '😀': 24, '😁': 25, '😂': 26, '😃': 27, '😄': 28, '😅': 29, '😆': 30, '😇': 31, '😈': 32,\
                    '😉': 33, '😊': 34, '😋': 35, '😌': 36, '😍': 37, '😎': 38, '😏': 39, '😐': 40, '😑': 41, '😒': 42, '😔': 43, '😕': 44, '😘': 45, '😜': 46, '😝': 47, '😞': 48,\
                    '😡': 49, '😢': 50, '😣': 51, '😤': 52, '😩': 53, '😪': 54, '😫': 55, '😬': 56, '😭': 57, '😱': 58, '😳': 59, '😴': 60, '🙈': 61, '🙌': 62, '🙏': 63}
    print('Load data...')
    df = load_data(text_path)  
    
    print('Preprocessing...')
    cleaned = clean_data(df) 
    text = get_text(cleaned) 
    labels = cleaned.label
    column_names = ['pre_punc', 'label_int']
    # define a new dataframe to save
    df_new = pd.DataFrame(columns = column_names)
    text_processor = TextProcessor()

    for i in tqdm(range(len(text))): 
        try:
            (pre_text, _, _) = preprocess(text[i], text_processor)
            df_new = df_new.append(pd.Series([pre_text, emoji_to_idx[labels[i]]], index=['pre_punc', 'label_int']), ignore_index=True)
        except: 
            continue
        
    # train-test split
    train, test = train_test_split(df_new, test_size = 0.2, stratify = df_new.label_int)
    val, test = train_test_split(df_new, test_size = 0.5, stratify = test.label_int)
    
    # save csv file
    train.to_csv('./dataset/train_Twitter.csv', index=False)
    val.to_csv('./dataset/val_Twitter.csv', index=False)
    test.to_csv('./dataset/test_Twitter.csv', index=False)

__main__()


