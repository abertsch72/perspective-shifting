from datasets import Dataset

def load_data(folder = "../data/gyafc/", dataset = "train"):
    inf = []
    form = []
    with open(folder + dataset + "-informal.txt") as f:
         inf = [line.strip() for line in f.readlines() if line.strip() != ""]
    with open(folder + dataset + "-formal.txt") as f:
         form = [line.strip() for line in f.readlines() if line.strip() != ""]
    labels = [0] * len(inf)
    labels.extend([1] * len(form))
    inf.extend(form)
    return {"text": inf, "label": labels}

def compute_metrics(eval_preds):
    metric = load_metric("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

raw_train = Dataset.from_dict(load_data(dataset="train")) 
raw_test = Dataset.from_dict(load_data(dataset="val"))

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

full_train_dataset = raw_train.map(tokenize_function, batched=True).shuffle(seed=42)
full_eval_dataset = raw_test.map(tokenize_function, batched=True).shuffle(seed=42)

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("../logs/gyafc/model", num_labels=2)

from transformers import TrainingArguments

training_args = TrainingArguments(output_dir="../logs/gyafc", per_device_train_batch_size = 4, num_train_epochs = 5,
    )

from transformers import Trainer

trainer = Trainer(model=model, args=training_args, train_dataset=full_train_dataset, eval_dataset=full_eval_dataset, compute_metrics = compute_metrics)

import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
trainer.evaluate()
#trainer.train()
#trainer.save_model("../logs/gyafc/model")
