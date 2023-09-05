from model import tokenizer, CommentSentimentModel, apply_model
from utils import import_random_test_data
from math import ceil
from torch import Tensor, load as loadModel

def print_page(page_num: int, data: list) -> None:
    for i in range(page_num*10, min(len(data), page_num*10+10)):
        print(f'{i}. {data[i]}')
    print(f'---- page {page_num} ----')

test_data = import_random_test_data()
model = loadModel('model1.pt')

page_num = 0
print('-'*30)
print(f'{ceil(len(test_data)/10)} pages of comments imported')
print_page(page_num, test_data)
print()
print('Enter the index of the comment to apply model, or \'<\'/\'>\' to jump between pages. Enter \'x\' to exit')

while True:
    op = input('input:')
    match op:
        case '>':
            index = -1
            page_num = 0 if page_num == len(test_data)-1 else page_num+1
            print_page(page_num, test_data)
        case '<':
            index = -1
            page_num = len(test_data)-1 if page_num == 0 else page_num-1
            print_page(page_num, test_data)
        case 'x':
            break
        case _:
            index = int(op)
    if index >= 0 and index < len(test_data):
        comment = test_data[index]
        print(f'Comment selected: {comment}')
        print(apply_model(comment, model))
        print()
