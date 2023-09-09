import sys
import os
import json
import openai
from typing import Union

sys.path.append(os.getcwd()+'\\')

from utils import *

def gpt_summarize(comments: list[str], aspect: str, config: dict[str, str], comment_repetition=3) -> list[dict[str, Union[list[str], str]]]:
    api_key = config['API_KEY']
    instruction = config['summarize_instruction'].format(ASPECTS_SUMMARY_ALIAS[aspect])

    openai.api_key = api_key
    comments_processed = list(map(lambda x: f'"{x}"', comments))

    summary_result = []

    for _ in range(comment_repetition):
        shuffle(comments_processed)
        clustered_comments = cluster_comments(comments_processed)
        
        for i in range(len(clustered_comments)):
            print(f'{i} / {len(clustered_comments)}')
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": instruction

                        },
                        {
                            "role": "user",
                            "content": f"[{clustered_comments[i]}]"
                        },
                    ]
                )

                resp: str = completion.choices[0].message.content
                print(resp)

                result = { 'source': clustered_comments[i].replace('"', '').split(',\n'), 'summary': resp }
                summary_result.append(result)
            except:
                print('request error')
    
    return summary_result

if __name__ == '__main__':
    config = get_config()
    
    pos_comments: dict[str, list[str]] = {}
    neg_comments: dict[str, list[str]] = {}
    for aspect in ASPECTS_SUMMARY_ALIAS.keys():
        with open(f'./data/summaryData/pos/{aspect}.json', mode='r') as file_json:
            pos_comments[aspect] = list(map(lambda metadata: metadata['comment'], json.load(file_json)))
        with open(f'./data/summaryData/neg/{aspect}.json', mode='r') as file_json:
            neg_comments[aspect] = list(map(lambda metadata: metadata['comment'], json.load(file_json)))
    
    
    for aspect in ASPECTS_SUMMARY_ALIAS.keys():
        summary_meta = gpt_summarize(pos_comments[aspect], aspect, config)
        with open(f'./data/summaryData/summaries/pos/{aspect}.json', mode='w+') as file_json:
            json.dump(summary_meta, file_json, indent=2)
        summary_meta = gpt_summarize(neg_comments[aspect], aspect, config)
        with open(f'./data/summaryData/summaries/neg/{aspect}.json', mode='w+') as file_json:
            json.dump(summary_meta, file_json, indent=2)
    

    

