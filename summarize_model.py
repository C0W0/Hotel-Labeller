from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from torch.utils.data import DataLoader
from utils import *

MODEL_NAME = 'gpt2'

tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(GPU)

def apply_model(comment: str, pre_model=model):
    print(f"Comment: \"{comment}\"")
    input_to_model = tokenizer.encode(comment, return_tensors='pt')
    result: torch.Tensor = pre_model.generate(input_to_model.to(GPU), max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50)
    return tokenizer.decode(result[0], skip_special_tokens=True)

sources: list[str] = []
summaries: list[str] = []
for sentiment in ['pos', 'neg']:
    for aspect in ASPECTS_SUMMARY_ALIAS.keys():
        with open(f'./data/summaryData/summaries/{sentiment}/{aspect}.json', mode='r') as file_json:
            test_data: list[dict] = json.load(file_json)
            sources.extend(list(map(lambda data: ' '.join(data['source']), test_data)))
            summaries.extend(list(map(lambda data: data['summary'], test_data)))

dataset = SummarizeDataSet(sources, summaries, tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(20):
    model.train()
    for (input, attention_mask), summary in dataloader:
        # Forward pass
        outputs = model(input, attention_mask=attention_mask, labels=summary)
        loss = outputs.loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}: Loss {loss.item()}")


torch.save(model.state_dict(), 'model_summarize.pt')
