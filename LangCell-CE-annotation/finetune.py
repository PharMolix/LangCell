# %%
# imports
from collections import Counter
import pickle
import subprocess

from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertForSequenceClassification
from transformers import Trainer
from transformers.training_args import TrainingArguments
import argparse
from utils import LangCellDataCollatorForCellClassification as DataCollatorForCellClassification
import os

# %% 
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default='/path/to/')
parser.add_argument("--data_path", type=str, default='/path/to/')
parser.add_argument("--output_path", type=str, default=None)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--test_size", type=float, default=0.33)
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()
model_path =  args.model_path
data_path = args.data_path
output_path = './output/' + model_path.split('/')[-1] + '/' + data_path.split('/')[-1].split('.')[0] + '/'
output_path = args.output_path if args.output_path else output_path
epochs = args.epochs
test_size = args.test_size

GPU_NUMBER = [args.device]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
os.environ["NCCL_DEBUG"] = "INFO"
# %%
dataset=load_from_disk(data_path)

trainset_organ_shuffled = dataset.shuffle(seed=1)
for label_name in ["celltype", "cell_type", "str_labels", "labels"]:
    if label_name in trainset_organ_shuffled.column_names:
        break
trainset_organ_shuffled = trainset_organ_shuffled.rename_column(label_name,"label")
target_names = list(Counter(trainset_organ_shuffled["label"]).keys())
target_name_id_dict = dict(zip(target_names,[i for i in range(len(target_names))]))

# change labels to numerical ids
def classes_to_ids(example):
    example["label"] = target_name_id_dict[example["label"]]
    return example
labeled_trainset = trainset_organ_shuffled.map(classes_to_ids, num_proc=16)
train_size = round(len(labeled_trainset)*(1-test_size))
labeled_train_split = labeled_trainset.select([i for i in range(0, train_size)])
labeled_eval_split = labeled_trainset.select([i for i in range(train_size, len(labeled_trainset))])
trained_labels = list(Counter(labeled_train_split["label"]).keys())

def if_trained_label(example):
    return example["label"] in trained_labels
labeled_eval_split_subset = labeled_eval_split.filter(if_trained_label, num_proc=16)

train_set = labeled_train_split
eval_set = labeled_eval_split_subset
label_name_id = target_name_id_dict
# %%
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy and macro f1 using sklearn's function
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    return {
      'accuracy': acc,
      'macro_f1': macro_f1
    }

# %%
# set model parameters
# max input size
max_input_size = 2 ** 11  # 2048
max_lr = 5e-5
freeze_layers = 0
num_gpus = 1
num_proc = 16
geneformer_batch_size = 16
lr_schedule_fn = "linear"
warmup_steps = 500
epochs = epochs
optimizer = "adamw"

# %%
import torch.nn as nn
organ_trainset = train_set
organ_evalset = eval_set
organ_label_dict = label_name_id

# set logging steps
logging_steps = round(len(organ_trainset)/geneformer_batch_size/2)

# reload pretrained model
model = BertForSequenceClassification.from_pretrained(model_path,
                                                    num_labels=len(organ_label_dict.keys()),
                                                    output_attentions = False,
                                                    output_hidden_states = False).cuda()

from geneformer import TranscriptomeTokenizer
tk = TranscriptomeTokenizer()
config = model.bert.config
if config.vocab_size == len(tk.gene_token_dict) - 1:
    embedding_layer = nn.Embedding(config.vocab_size + 1, config.hidden_size, padding_idx=config.pad_token_id)
    for param, param_pretrain in zip(embedding_layer.parameters(), model.bert.embeddings.word_embeddings.parameters()):
        param.data[:-1] = param_pretrain.data
    model.bert.embeddings.word_embeddings = embedding_layer
elif config.vocab_size != len(tk.gene_token_dict):
    raise Exception("Vocab size does not match.")


# define output directory path
output_dir = output_path

# ensure not overwriting previously saved model
saved_model_test = os.path.join(output_dir, f"pytorch_model.bin")
if os.path.isfile(saved_model_test) == True:
    raise Exception("Model already saved to this directory.")

# make output directory
subprocess.call(f'mkdir {output_dir}', shell=True)

# set training arguments
training_args = {
    "learning_rate": max_lr,
    "do_train": True,
    "do_eval": True,
    "evaluation_strategy": "epoch",
    "save_total_limit": 1,
    "logging_steps": logging_steps,
    "group_by_length": True,
    "length_column_name": "length",
    "disable_tqdm": False,
    "lr_scheduler_type": lr_schedule_fn,
    "warmup_steps": warmup_steps,
    "weight_decay": 0.001,
    "per_device_train_batch_size": geneformer_batch_size,
    "per_device_eval_batch_size": geneformer_batch_size,
    "num_train_epochs": epochs,
    "load_best_model_at_end": False,
    "output_dir": output_dir,
}

training_args_init = TrainingArguments(**training_args)

# create the trainer
trainer = Trainer(
    model=model,
    args=training_args_init,
    data_collator=DataCollatorForCellClassification(),
    train_dataset=organ_trainset,
    eval_dataset=organ_evalset,
    compute_metrics=compute_metrics
)
# train the cell type classifier

# %%
trainer.train()
predictions = trainer.predict(organ_evalset)
with open(f"{output_dir}predictions.pickle", "wb") as fp:
    pickle.dump(predictions, fp)
trainer.save_metrics("eval",predictions.metrics)
trainer.save_model(output_dir)


