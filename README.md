# MultiEmo 

This repository contains the dataset and the code for the paper **MultiEmo: multi-task framework for emoji prediction**. (*add publisher)
The repository is structured as follows:

``` 
.
├── Dataset 
│   ├── data_loader.py 
│   ├── preprocessing.py 
│   ├── GoEmotion.csv
│   └── Twitter.csv 
├── Models
│   ├── MultiEmo.py 
│   └── README.md
├── run_multiemo.py
├── README.md
└── requirements.txt
``` 


Data Format
-------------

#### (1) Twitter dataset
Our experiment employed two types of dataset.
First one is the twitter emoji dataset which we get using REST API. For considering the imbalance in the actual usage of emoji on social media, we only considered the top 64 emojis. These are the top 64 emojis which we get from [emojitracker](http://www.emojitracker.com/).


Twitter dataset can be found in ``./data/Twitter.csv``. Each row represents a twitter post including at least one emoji of the top 60 emojis. This file includes the following columns:
* post
* emoji 

The data we used for trainind the models includes posts with only one emoji. Also, we filtered the post data ~. You can get this by executing the following script.
```
python ./data/preprocessing.py --Twitter.csv
``` 

#### (2) GoEmotion dataset
For emotion detection, we employed GoEmotion which was released [here](https://github.com/google-research/google-research/tree/master/goemotions).
GoEmotion is a dataset labeled 58,000 Reddit  comments with 28 emotions. Furthermore, all the comments were also labeled with hierarchical grouping (positive, negative, ambiguous + neutral) and Ekman emotion (anger, disgust, fear, joy, sadness, surprise + neutral). To exclude ambiguous data as much as possible, we removed all the comments labeled as neutral. Finally, this file includes the following columns:
* text
* emotion
* grouping
* Ekman

Also, we used this dataset after going through our own preprocessing pipeline. You can do this by,
```
python ./data/preprocessing.py --GoEmotion.csv
```


Model
-------------
Here is the architecture of MultiEmo. 


<img src="https://user-images.githubusercontent.com/63252403/148646373-28f826ff-ed44-4129-963c-3f08a93ca686.JPG" width="700" height="400"/> 



For MultiEmo, you can train model by following script.
```
python run_multiemo.py --batch_size 16 --glove_path --type 3
```
Since there are one type of single emoji classifier and three types of multi-task emoji classifiers which we call "MultiEmo", you can clarify the type of model you want to train by varying the argument "mode".
* 1: single-task 
* 2-a: MultiEmo with emo_a 
* 2-b: MultiEmo with emo_a 
* 2-c: MultiEmo with emo_a 
* 3-a: MultiEmo with emo_a 
* 3-b: MultiEmo with emo_a 
* 3-c: MultiEmo with emo_a 


Requirements 
-------------
For experimental setup, ``requirements.txt`` lists down the requirements for running the code on the repository. Note that a cuda device is required.
The requirements can be downloaded using,
```
pip install -r requirements.txt
``` 


Contact 
-------------
For more details on the design and content of our model, please see [our paper]().
If you have any questions regarding the code, please create an issue or contact the [owner]() of this repository.

If you use our work, please cite us:

