import json
import random
import os
from torch.utils.data import Dataset
from torch import Tensor, device
import numpy as np

_comments_to_ignore = set(['No Positive', 'No Negative', 'All', 'Everything', 'everything', 'Nothing', 'nothing'])
ASPECTS = [
    'Cleanliness',
    'Food',
    'Service and Staff',
    'Amenities',
    'Location',
    'Room Comfort',
    'Parking',
    'Security',
]
GPU = device(0)

def should_ignore(comment: str) -> bool:
    return comment in _comments_to_ignore


def pick_random(arr: list[any]) -> any:
    return arr[random.randint(0, len(arr)-1)]


def _shuffle(arr: list[any]) -> None:
    arr_len = len(arr)
    for i in range(arr_len):
        dest_index = random.randint(0, arr_len-1)
        arr[i], arr[dest_index] = arr[dest_index], arr[i]


def aspect_to_vec(labels: dict[str, float]) -> list[float]:
    result_vec: list[float] = []
    for aspect in ASPECTS:
        result_vec.append(labels[aspect])
    return result_vec

def vec_to_aspect(aspect_vec: list[float]) -> dict[str, float]:
    result_aspect = {}
    for i in range(len(aspect_vec)):
        result_aspect[ASPECTS[i]] = aspect_vec[i]
    return result_aspect


def import_all_training_data() -> tuple[list[str], np.ndarray]:
    reviews: list[dict[str, any]] = []
    with open('./data/labelled/pos/labelled.json', 'r') as pos_json:
        reviews.extend(json.load(pos_json))
    with open('./data/labelled/neg/labelled.json', 'r') as neg_json:
        reviews.extend(json.load(neg_json))
    _shuffle(reviews)

    comments: list[str] = []
    label_vecs: list[list[float]] = []
    for labelled_entry in reviews:
        comments.append(labelled_entry['comment'])
        label_vecs.append(aspect_to_vec(labelled_entry['labels']))
    
    return (comments, np.array(label_vecs))


def import_random_test_data() -> list[str]:
    data_dir = os.path.join(os.getcwd(), 'data')
    list_of_positives = os.listdir(os.path.join(data_dir, 'positive'))
    list_of_negatives = os.listdir(os.path.join(data_dir, 'negative'))
    positive_path = os.path.join(data_dir, 'positive', pick_random(list_of_positives))
    negative_path = os.path.join(data_dir, 'negative', pick_random(list_of_negatives))

    comments: list[str] = []
    with open(positive_path) as pos_json:
        for comment in json.load(pos_json):
            if should_ignore(comment): continue
            comments.append(comment)
    with open(negative_path) as neg_json:
        for comment in json.load(neg_json):
            if should_ignore(comment): continue
            comments.append(comment)
    _shuffle(comments)

    return comments

class CommentDataSet(Dataset):
    def __init__(self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor) -> None:
        super().__init__()

        self.x = input_ids.to(GPU)
        self.masks = attention_mask.to(GPU)
        self.y = labels.to(GPU)
        
        self.n_sample = len(labels)

    def __getitem__(self, index) -> tuple[tuple[Tensor], Tensor]:
        return (self.x[index], self.masks[index]), self.y[index]
    
    def __len__(self):
        return self.n_sample