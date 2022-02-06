# MultiEmo 

This repository contains the dataset and the implementation code for the paper **MultiEmo: multi-task framework for emoji prediction**.

Overview🧐
-------------
* [data/](https://github.com/sange1104/MultiEmo/tree/main/data) contains raw and preprocessed datasets for train, validation and test. preprocessing.py would help preprocess the raw text. Also, pre-trained model weight and vocabulary for tokenizing will be located here. 
* [scripts/](https://github.com/sange1104/MultiEmo/tree/main/scripts) contains code for implementing the model and thus reproducing results in the paper.
* [checkpoints/](https://github.com/sange1104/MultiEmo/tree/main/checkpoints) is a repository where a checkpoint of the trained model such as weight information or optimizer state would be saved.
* README.md
* requirements.txt


Setup
-------------
**Environment setup**

For experimental setup, ``requirements.txt`` lists down the requirements for running the code on the repository. Note that a cuda device is required.
The requirements can be downloaded using,
```
pip install -r requirements.txt
``` 

**Training setup**

You can download the model checkpoints from the [TorchMoji repo](https://github.com/huggingface/torchMoji). We also employed tokenizer used in torchmoji, you should download this vocabulary [here](https://github.com/huggingface/torchMoji/blob/master/model/vocabulary.json). Both pre-trained weights and vocabulary file is expected to be located in [data/](https://github.com/sange1104/MultiEmo/tree/main/data).

Data Format
-------------

#### (1) Twitter dataset
Our experiment needs two dataset for employing different task.

First one is the twitter emoji dataset which we get using REST API. For considering the imbalance in the actual usage of emoji on social media, we only considered the top 64 emojis. 
```
✅✨🌚🎉🎶👀👇👌👍👏👑💀💔💕💖💗💙💚💛💜💞💪💯🔥😀😁😂😃😄😅😆😇😈😉😊😋😌😍😎😏😐😑😒😔😕😘😜😝😞😡😢😣😤😩😪😫😬😭😱😳😴🙈🙌🙏
``` 
These are the top 64 emojis which we get from [emojitracker](http://www.emojitracker.com/).


Twitter dataset can be found in ``./data/Twitter.csv``. Each row represents a twitter post including at least one emoji of the top 64 emojis. 
The data we used for training includes posts with only one emoji. You can preprocess the raw data by running the following script. After running the ``./data/preprocessing.py``, preprocessed dataset will split into 3 files for train, validation, and test, respectively.
```
python ./data/preprocessing.py --Twitter.csv
``` 

#### (2) GoEmotion dataset
For emotion detection, we employed GoEmotion which was released [here](https://github.com/google-research/google-research/tree/master/goemotions).
GoEmotion is a dataset labeled 58,000 Reddit  comments with 28 emotions. Furthermore, all the comments were also labeled with hierarchical grouping (positive, negative, ambiguous + neutral) and Ekman emotion (anger, disgust, fear, joy, sadness, surprise + neutral). To exclude ambiguous data as much as possible, we removed all the comments labeled as neutral during training stage. Also, we used this dataset after employing the same preprocessing pipeline we used for Twitter dataset.


How to train
-------------
You can run [train.py](https://github.com/sange1104/MultiEmo/blob/main/scripts/train.py) setting arguments as follows:
|Name|Required|Type|Default|Options|
|---|---|---|---|---|
|**aux_num**|Yes|int|-|1,2,3|
|**aux_task**|Yes|str|-|'emo', 'emo sent'|
|gpu_num|Yes|int|-|1,2|
|learning_rate|No|float|1e-4|-|
|batch_size|No|int|64|-|
|num_epoch|No|int|50|-|
|save_cp|No|bool|True|-| 
|patience|No|int|0|-|
|early_stop|No|int|2|-|
|decay|No|bool|False|-|
|fine_tuning|No|bool|False|-|
|pre_trained|No|bool|True|-|


Since there are one type of single emoji classifier and three types of multi-task classifiers which we call "MultiEmo", you can clarify the type of model you want to train by varying the argument "aux_task".
Options of aux_task can be one of,

* emo: emotion detection labeled for 27 emotions
* Ekman: emotion detection labeled for 6 Ekman emotions
* sent: emotion detection labeled for 3 sentiments

If you want a multi-task model with more than 1 auxiliary task, you can give several tasks as follows:
```
python ./scripts/train.py --aux_num 2 --aux_task emo sent --gpu_num 1
``` 
Note that the number of aux_task and aux_num should be equal.


Simple Demo
-------------
![readme](https://user-images.githubusercontent.com/63252403/152666369-9e0cbd97-5c41-4264-8562-5a72499a1cef.JPG)

You can test our model with this simple demo. You can do this by, 
```
python run_multiemo.py 
``` 




