from transformers import PegasusTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, PegasusForConditionalGeneration
import torch
import evaluate
from utils import *

rouge = evaluate.load('rouge')

MODEL_NAME = 'google/pegasus-cnn_dailymail'

tokenizer: PegasusTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model: PegasusForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(GPU)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

def apply_model(comment: str, pre_model=model):
    print(f"Comment: \"{comment}\"")
    input_to_model = tokenizer.encode(comment, return_tensors='pt')
    result: torch.Tensor = pre_model.generate(input_to_model.to(GPU), max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50)
    return tokenizer.decode(result[0], skip_special_tokens=True)

imported_data: list[dict] = []
for sentiment in ['pos', 'neg']:
    for aspect in ASPECTS_SUMMARY_ALIAS.keys():
        with open(f'./data/summaryData/summaries/{sentiment}/{aspect}.json', mode='r') as file_json:
            imported_data.extend(json.load(file_json))
        break
shuffle(imported_data)
(test_data, train_data) = partition_data(imported_data, 0.1)

sources_test: list[str] = list(map(lambda data: ' '.join(data['source']), test_data))
summaries_test: list[str] = list(map(lambda data: data['summary'], test_data))
dataset_test = SummarizeDataSet(sources_test, summaries_test, tokenizer)

sources_train: list[str] = list(map(lambda data: ' '.join(data['source']), train_data))
summaries_train: list[str] = list(map(lambda data: data['summary'], train_data))
dataset_train = SummarizeDataSet(sources_train, summaries_train, tokenizer)

training_args = Seq2SeqTrainingArguments(
    output_dir="./",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=5,
    per_device_eval_batch_size=5,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=8,
    predict_with_generate=True,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model('./summary_model_01')
