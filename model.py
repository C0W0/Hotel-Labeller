from transformers import BertModel, BertTokenizer
import torch
from torch.utils.data import DataLoader
from utils import import_all_training_data, import_random_test_data, pick_random, vec_to_aspect, CommentDataSet, GPU

class CommentSentimentModel(torch.nn.Module):
    def __init__(self) -> None:
        super(CommentSentimentModel, self).__init__()
        self.bert: BertModel = BertModel.from_pretrained(MODEL_NAME, num_labels=8).to(GPU)
        self.aspect_output = torch.nn.Linear(self.bert.config.hidden_size, 8, device=GPU)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        aspect_scores: torch.Tensor = self.aspect_output(bert_output)
        aspect_scores = (torch.selu(aspect_scores)+1.5)/3
        return aspect_scores

MODEL_NAME = 'bert-base-uncased'

model = CommentSentimentModel()
tokenizer: BertTokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def apply_model(comment: str):
    print(f"Comment: \"{comment}\"")
    input_to_model = tokenizer.encode_plus(
        comment,
        max_length=256, # average comment length: 103
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        is_split_into_words=True,
    )
    result: torch.Tensor = model(input_to_model.input_ids.to(GPU), input_to_model.attention_mask.to(GPU))[0]
    print(vec_to_aspect(result.tolist()))
    print()

def train():
    train_comments, train_labels = import_all_training_data()
    inputs1 = tokenizer.batch_encode_plus(
        train_comments,
        max_length=256, # average comment length: 103
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        is_split_into_words=True,
    )

    dataset = CommentDataSet(inputs1.input_ids, inputs1.attention_mask, torch.from_numpy(train_labels).float())
    dataloader = DataLoader(dataset, batch_size=8)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(50):
        for (input_ids, attention_mask), label in dataloader:
            aspect_scores = model(input_ids, attention_mask)
            loss = loss_fn(aspect_scores, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print(f'{epoch} / 50')


train()
# testing
test_data = import_random_test_data()
for _ in range(10):
    apply_model(pick_random(test_data))

# storage
torch.save(model, 'model1.pt')