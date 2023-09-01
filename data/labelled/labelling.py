import json
import os
import openai
from utils import should_ignore

def label(api_key: str, label_folder: str, main_instruction: str):
    label_dir = os.path.join(os.getcwd(), 'data', 'labelled', label_folder)
    comment_list: list[str] = []

    for file_name in os.listdir(label_dir):
        if file_name == 'labelled.json': continue
        with open(os.path.join(label_dir, file_name), 'r') as comments_json:
            for comment in json.load(comments_json):
                if should_ignore(comment): continue
                comment_list.append(f"\"{comment}\"")

    comment_clusters: list[str] = []
    current_cluster: list[str] = []
    curr_cluster_str_len = 0
    for comment in comment_list:
        current_cluster.append(comment)
        curr_cluster_str_len += len(comment)

        if curr_cluster_str_len >= 2500 or len(current_cluster) >= 10:
            comment_clusters.append(',\n'.join(current_cluster))
            current_cluster.clear()
            curr_cluster_str_len = 0
    
    if curr_cluster_str_len != 0:
        comment_clusters.append(',\n'.join(current_cluster))
        current_cluster.clear()
        curr_cluster_str_len = 0

    # for clustered in comment_clusters:
    #     print(clustered)
    #     print()

    specific_instructions = {'pos': 'The comments are mostly positive, so treat ambiguous sentiments as positive', 'neg': 'The comments are mostly negative, so treat ambiguous sentiments as negative'}
    if label_folder in specific_instructions:
        main_instruction += f'\n{specific_instructions[label_folder]}'

    openai.api_key = api_key
    decoder = json.JSONDecoder()
    labelled_resp = []

    for i in range(len(comment_clusters)):
        print(f'{i} / {len(comment_clusters)}')

        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": main_instruction

                    },
                    {
                        "role": "user",
                        "content": comment_clusters[i]
                    },
                ]
            )

            resp: str = completion.choices[0].message.content
            labelled_resp.extend(decoder.decode(resp))

            print(resp)
        except:
            print('request failed')
            continue
    
    with open(f'{label_dir}/labelled.json', 'w+') as json_out:
        json.dump(labelled_resp, json_out)


if __name__ == '__main__':
    config: dict[str, str]
    with open('./config.json', mode='r') as config_file:
        config = json.load(config_file)
    label(api_key=config['API_KEY'], main_instruction=config['gpt_instruction'], label_folder='neg')