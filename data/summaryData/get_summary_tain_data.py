import sys
import os
import json
from typing import Union

sys.path.append(os.getcwd()+'\\')

from utils import import_random_test_data, ASPECTS, ASPECTS_SUMMARY_ALIAS, should_ignore
from data.labelled.labelling import label

def label_using_gpt(config: dict[str, str]):
    test_data = import_random_test_data()

    pos_sentiments: dict[str, list[dict[str, Union[float, str]]]] = {}
    neg_sentiments: dict[str, list[dict[str, Union[float, str]]]] = {}
    for aspect in ASPECTS:
        pos_sentiments[aspect] = []
        neg_sentiments[aspect] = []

    label_result = label(api_key=config['API_KEY'], comment_list=test_data, main_instruction=config['label_instruction'])

    for gpt_resp in label_result:
        try:
            comment: str = gpt_resp['comment']
            for (aspect, score) in gpt_resp['labels'].items():
                sentiment_list: dict[str, list[dict[str, Union[float, str]]]]
                if score > 0.65:
                    sentiment_list = pos_sentiments
                elif score < 0.35:
                    sentiment_list = neg_sentiments
                else:
                    continue

                comment_data = {}
                comment_data['comment'] = comment
                comment_data['score'] = score
                sentiment_list[aspect].append(comment_data)
        except:
            continue

    for (aspect, meta_data) in pos_sentiments.items():
        with open(os.path.join('./data/summaryData/pos', f'{aspect}.json'), mode='w+') as file_json:
            json.dump(meta_data, file_json, indent=2)
    for (aspect, meta_data) in neg_sentiments.items():
        with open(os.path.join('./data/summaryData/neg', f'{aspect}.json'), mode='w+') as file_json:
            json.dump(meta_data, file_json, indent=2)

def use_existing_training_data():
    all_training_data: list[dict[str, Union[str, dict[str, float]]]] = {}
    with open('./data/labelled/neg/labelled.json') as labelled_file:
        all_training_data = json.load(labelled_file)
    with open('./data/labelled/pos/labelled.json') as labelled_file:
        all_training_data.extend(json.load(labelled_file))
    
    pos_sentiments: dict[str, list[dict[str, Union[float, str]]]] = {}
    neg_sentiments: dict[str, list[dict[str, Union[float, str]]]] = {}

    
    for aspect in ASPECTS_SUMMARY_ALIAS.keys():
        pos_sentiments[aspect] = []
        neg_sentiments[aspect] = []

    for comment_data in all_training_data:
        comment: str = comment_data['comment']
        for aspect in ASPECTS_SUMMARY_ALIAS.keys():
            score = comment_data['labels'][aspect]
            if score > 0.65:
                pos_sentiments[aspect].append({ 'comment': comment,  'score': score })
            if score < 0.35:
                neg_sentiments[aspect].append({ 'comment': comment,  'score': score })
            else:
                continue

    for aspect in ASPECTS_SUMMARY_ALIAS.keys():
        with open(f'./data/summaryData/pos/{aspect}.json', mode='r') as file_json:
            pos_sentiments[aspect].extend(json.load(file_json))
        with open(f'./data/summaryData/neg/{aspect}.json', mode='r') as file_json:
            neg_sentiments[aspect].extend(json.load(file_json))
    
    for aspect in ASPECTS_SUMMARY_ALIAS.keys():
        with open(f'./data/summaryData/pos/{aspect}.json', mode='w+') as file_json:
            json.dump(pos_sentiments[aspect], file_json, indent=2)
        with open(f'./data/summaryData/neg/{aspect}.json', mode='w+') as file_json:
            json.dump(neg_sentiments[aspect], file_json, indent=2)


# config: dict[str, str]
# with open('./config.json', mode='r') as config_file:
#     config = json.load(config_file)
use_existing_training_data()