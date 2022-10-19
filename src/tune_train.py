"""
originally based on https://github.com/elsanns/xai-nlp-notebooks/blob/master/fine_tune_bart_summarization_two_langs.ipynb
"""

import numpy as np
import datasets
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import nltk
import wandb

import argparse
from typing import Text


from load_data import load_data 

parser = argparse.ArgumentParser(description='Arguments to train model')
parser.add_argument('--model_name', type=Text, default="facebook/bart-base",
                    help='name of the model to train')

parser.add_argument('--model_dir', type=Text, default=None,
                    help='place to save checkpoints and models')

parser.add_argument('--output_dir', type=Text, default=None,
                    help='place to save output files')

parser.add_argument('--encoder_max_len', type=int, default=512,
                    help='max input length')

parser.add_argument('--generation_max_len', type=int, default=90,
                    help='max output length')

parser.add_argument('--num_epochs', type=int, default=3,
                    help='max number of epochs to run')

parser.add_argument('--batch_size', type=int, default=16,
                    help='the batch size')

parser.add_argument('--seed', type=int, default=1,
                    help='seed for nondeterminism')

parser.add_argument('--patience', type=int, default=3,
                    help='patience for early stopping')

parser.add_argument('--weight_decay', type=float, default=0.1,
                    help='weight decay hyperparameter')

parser.add_argument('--label_smoothing', type=float, default=0.0,
                    help='label smoothing hyperparameter')

parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help='grad accumulation steps hyperparameter')

parser.add_argument('--learning_rate', type=float, default=5e-5,
                    help='learning rate')

parser.add_argument('--wandb', type=bool, default=True,
                    help='whether to run wandb logging')

parser.add_argument('--do_train', action='store_true', default=False,
                    help='whether to run training')

parser.add_argument('--do_tune', action='store_true', default=False,
                    help='whether to run tuning')

parser.add_argument('--do_eval', action='store_true', default=False,
                    help='whether to run eval')

parser.add_argument('--do_eval_test', action='store_true', default=False,
                    help='whether to eval over test at the end')

parser.add_argument('--do_gen_val', action='store_true', default=False,
                    help='whether to generate over val')

parser.add_argument('--do_gen_test', action='store_true', default=False,
                    help='whether to generate over test')

parser.add_argument('--save_model', action='store_true', default=False,
                    help='whether to save model')

parser.add_argument('--project', type=Text, default="perspective-shift",
                    help='the wandb project to log into')

parser.add_argument('--group', type=Text, default="",
                    help='the wandb project group to log into')

parser.add_argument('--name', type=Text, default="",
                    help='a name for the wandb run')

parser.add_argument('--start', type=int, default=0, help='start index of val')
parser.add_argument('--end', type=int, default=-1, help='end index of val')

parser.add_argument('--start_train', type=int, default=0, help='start index of train')
parser.add_argument('--end_train', type=int, default=-1, help='end index of train')

parser.add_argument('--dataset', type=Text, choices=["lbl", "convo", "mask", "nocontext", "gyafc", "heuristic"],
                    help='data to train and evaluate on')
parser.add_argument('--dataset_split', type=Text, choices=["PStrain", "GENtrain", "GENval", "GENtest"], default="PStrain",
                    help='split of dataset to use')

args = parser.parse_args()

if args.do_tune:
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.suggest.hyperopt import HyperOptSearch

model_dir = args.model_dir if args.model_dir is not None else "models/" + args.group + "/" + args.name
output_dir = args.output_dir if args.output_dir is not None else "outputs/" + args.group + "/" + args.name


if args.wandb:
    wandb.init(name=args.name, group=args.group, entity="gormleylab", project=args.project)


def model_init():
    return AutoModelForSeq2SeqLM.from_pretrained(args.model_name, return_dict=False)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

train_data, val_data, test_data = load_data(args.dataset, args.dataset_split)
   

print(train_data["original"][65:70])
print(train_data["shifted"][65:70])

train_data["original"] = train_data["original"][args.start_train:args.end_train]
train_data["shifted"] = train_data["shifted"][args.start_train:args.end_train]

val_data["original"] = val_data["original"][args.start:args.end]
val_data["shifted"] = val_data["shifted"][args.start:args.end]
nltk.download("punkt", quiet=True)
metric = datasets.load_metric("rouge")

train_data_txt = datasets.Dataset.from_dict(train_data)
validation_data_txt = datasets.Dataset.from_dict(val_data)
test_data_txt = datasets.Dataset.from_dict(test_data)

NUM_EPOCHS = args.num_epochs


def batch_tokenize_preprocess(batch, tokenizer, max_source_length, max_target_length, train=False):
    source, target = batch['original'], batch['shifted']

    source_tokenized = tokenizer(
        source, padding="max_length", truncation=True, max_length=max_source_length
    )
    target_tokenized = tokenizer(
        target, padding="max_length", truncation=True, max_length=max_target_length
    )

    batch = {k: v for k, v in source_tokenized.items()}
    # Ignore padding in the loss
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ]
    return batch


train_data = train_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, args.encoder_max_len, args.generation_max_len, train=True
    ),
    batched=True,
    remove_columns=train_data_txt.column_names,

)

validation_data = validation_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, args.encoder_max_len, args.generation_max_len,
    ),
    batched=True,
    remove_columns=validation_data_txt.column_names,
)

test_data = test_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, args.encoder_max_len, args.generation_max_len,
    ),
    batched=True,
    remove_columns=test_data_txt.column_names,
)



def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_objective(metrics):
    #_, _, results = metrics    
    results = metrics
    return results["eval_rouge2"]



def compute_metrics(eval_preds):
    print(eval_preds)
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    result["eval_rouge"] = result["rouge1"]
    print(result.keys())
    return result



training_args = Seq2SeqTrainingArguments(
    output_dir=model_dir,
    seed=args.seed,
    #fp16=True,
    num_train_epochs=NUM_EPOCHS,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    label_smoothing_factor=args.label_smoothing,
    weight_decay=args.weight_decay,
    predict_with_generate=True,
    load_best_model_at_end=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="rouge",
)

data_collator = DataCollatorForSeq2Seq(tokenizer)

trainer = Seq2SeqTrainer(
        model_init=model_init,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=validation_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience = args.patience)],
)

batch_options = [option for option in list(range(1, args.batch_size + 1)) if (option == 1 or option == 2 or option % 4 == 0)]
print(batch_options)
if args.do_tune:
    tune_config = {
         "num_train_epochs": tune.choice([5, 7, 10, 15, 20, 25]),
         "per_device_train_batch_size": tune.choice(batch_options),
         "per_device_eval_batch_size": tune.choice(batch_options),
         "learning_rate": tune.loguniform(1e-6, 1e-3),
         "weight_decay": tune.uniform(0, args.weight_decay * 2),
         "gradient_accumulation_steps": tune.choice([1, 2, 4, 8, 16, 24]),
         "label_smoothing_factor": tune.uniform(0, 0.2),
    }

    current_best_params = [{
         "num_train_epochs": 15,
         "per_device_train_batch_size": args.batch_size,
         "per_device_eval_batch_size": args.batch_size,
         "gradient_accumulation_steps": 4,
         "learning_rate": args.learning_rate,
         "weight_decay": args.weight_decay,
        "label_smoothing_factor": 0.0,
     }]

    trainer.hyperparameter_search(
        hp_space=lambda _: tune_config,
        direction="maximize",
        backend="ray",
        compute_objective=compute_objective,
        n_trials=50,
        resources_per_trial={"gpu": 1},
        local_dir = "./ray_out",
        search_alg=HyperOptSearch(metric="objective", mode="max", points_to_evaluate=current_best_params),
        scheduler=ASHAScheduler(metric="objective", mode="max"),
    )


"""Train the model"""

if args.do_train:
    trainer.train()

"""Evaluate after fine-tuning"""

if args.do_eval:
    trainer.evaluate(max_length=args.generation_max_len)

if args.do_eval_test:
    print("Test data:")
    trainer.evaluate(test_data, max_length=args.generation_max_len)

if args.do_train and args.save_model:
    trainer.save_model(model_dir + "/final")

def generate_summary(test_samples, trainer):
    inputs = tokenizer(
        test_samples["original"],
        padding="max_length",
        truncation=True,
        max_length=args.encoder_max_len,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(trainer.model.device)
    attention_mask = inputs.attention_mask.to(trainer.model.device)
    outputs = trainer.model.generate(input_ids, attention_mask=attention_mask, max_length=args.generation_max_len)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs, output_str

if args.do_gen_val:
    test_samples = validation_data_txt
elif args.do_gen_test:
    test_samples = test_data_txt
else:
    test_samples = []
p=0

with open(output_dir + ".txt", "a") as f:
    for i in range(args.batch_size + 1, len(test_samples) + 1, args.batch_size):
        summaries_after_tuning = generate_summary(test_samples.select(range(p, i)), trainer)[1]
        p = i
        for summ in summaries_after_tuning:
            print(summ)
            f.write(summ + "\n")
        print(p)
    if p < len(test_samples):
        summaries_after_tuning = generate_summary(test_samples.select(range(p, len(test_samples))), trainer)[1]
        for summ in summaries_after_tuning:
            print(summ)
            f.write(summ + "\n")
