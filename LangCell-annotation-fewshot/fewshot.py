# %%
import os
from datasets import load_from_disk
import torch.nn as nn, torch.nn.functional as F
import torch, json
from transformers import BertTokenizer, BertModel
from utils import BertModel as MedBertModel
from utils import LangCellDataCollatorForCellClassification as DataCollatorForCellClassification
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
import subprocess
from sklearn.metrics import accuracy_score, f1_score

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="/path/to/")
parser.add_argument("--output_path", type=str, default=None)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--train_batchsize", type=int, default=12)
parser.add_argument("--test_batchsize", type=int, default=64)
parser.add_argument("--nshot", type=int, default=1)
parser.add_argument("--seed", type=int, default=2024)
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()
# model_path =  args.model_path
data_path = args.data_path
epochs = args.epochs
train_batchsize = args.train_batchsize
test_batchsize = args.test_batchsize
seed = args.seed
nshot = args.nshot
output_path = 'output/ctm_' + str(nshot) + '-shot/'
output_path = args.output_path if args.output_path else output_path
subprocess.call(f'mkdir {output_path}', shell=True)

GPU_NUMBER = [args.device]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])

# %%
class Pooler(nn.Module):
    def __init__(self, config, pretrained_proj, proj_dim):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, proj_dim)
        self.proj.load_state_dict(torch.load(pretrained_proj))
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled_output = hidden_states[:, 0]
        pooled_output = F.normalize(self.proj(pooled_output), dim=-1)
        return pooled_output
    
model = BertModel.from_pretrained('/path/to/')
model.pooler = Pooler(model.config, pretrained_proj='/path/to/', proj_dim=256)
proj = model.pooler.proj
# model = model.module
model = model.to("cuda")

text_pretrained_model = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
tokenizer = BertTokenizer.from_pretrained(text_pretrained_model)
tokenizer.add_special_tokens({'bos_token':'[DEC]'})
tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
text_encoder = MedBertModel.from_pretrained('/path/to/', add_pooling_layer=True)
text_encoder.pooler = Pooler(text_encoder.config, pretrained_proj='/path/to/', proj_dim=256)
text_encoder = text_encoder.to("cuda")

ctm_head = nn.Linear(text_encoder.config.hidden_size, 2)
ctm_head.load_state_dict(torch.load('/path/to/'))
ctm_head = ctm_head.to("cuda")

def text_encode(text):
    text = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
    text = text_encoder(**text).pooler_output
    # text = F.normalize(model.text_projector(text))
    return text

def cell_encode(cell_input_ids, cell_atts):
    cell = model(cell_input_ids.to("cuda"), cell_atts.to("cuda"))
    cell_last_h = cell.last_hidden_state
    cell_pooler = cell.pooler_output
    return cell_last_h, cell_pooler

def ctm(text, cell_emb, cell_atts):
    # n texts, n cells -> n scores
    text = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
    output = text_encoder(**text,
                encoder_hidden_states = cell_emb.to("cuda"),
                encoder_attention_mask = cell_atts.to("cuda"),
                return_dict = True,
                mode = 'multimodal',
                )
    logits = ctm_head(output.last_hidden_state[:, 0, :])
    logits = F.softmax(logits, dim=-1)[..., 1] # [n]
    return logits

# %%
dataset = load_from_disk(data_path)
dataset_sub = dataset.shuffle(seed)#.select(range(300))
for label_name in ["celltype", "cell_type", "str_labels", "labels"]:
    if label_name in dataset_sub.column_names:
        break
if label_name != "celltype":
    dataset_sub = dataset_sub.rename_column(label_name,"celltype")
dataset_sub = dataset_sub.filter(lambda example: example['celltype'] != 'Other')

ontology = json.load(open('/path/to/'))
name2id = {ontology[id]['name']: id for id in ontology}
def gettextfromname(name, discription=True):
    ontology_name = name.lower()
    id = name2id[ontology_name.lower()]
    s = "cell type: "
    s += ontology[id]['name'] + '. ' 
    if discription:
        if ontology[id]['def'] != []:
            s += ontology[id]['def'] + '; '
    return s

types = list(set(dataset_sub['celltype']))
texts = [gettextfromname(typename) for typename in types]
type2num = dict([(type, i) for i, type in enumerate(types)])

# %%
def classes_to_ids(example):
    example["label"] = type2num[example["celltype"]]
    return example
dataset = dataset_sub.map(classes_to_ids, num_proc=16)
dataset = dataset.remove_columns(['celltype', 'length'])

# split
label_num = len(type2num.keys())
type2trainlist = {}
for i in range(label_num):
    type2trainlist[i] = []
if nshot >= 1:
    for i, l in enumerate(dataset["label"]):
        if len(type2trainlist[l]) < nshot:
            type2trainlist[l].append(i)
            br = True
            for k in type2trainlist.keys():
                if len(type2trainlist[k]) < nshot:
                    br = False
                    break
            if br:
                break
train_idx = []
for k in type2trainlist.keys():
    train_idx += type2trainlist[k]
test_idx = list(set(range(len(dataset))) - set(train_idx))

traindataset = dataset.select(train_idx).shuffle(seed)
testdataset = dataset.select(test_idx).shuffle(seed)
# traindataset, train_ind = extract_data_based_on_class(dataset, train_cls)
# testdataset, test_ind = extract_data_based_on_class(dataset, test_cls)

# train_batchsize, test_batchsize = 4, 64
eval_num = 500
train_loader = DataLoader(traindataset, batch_size=train_batchsize, 
                          collate_fn=DataCollatorForCellClassification(), shuffle=False)
test_loader = DataLoader(testdataset, batch_size=test_batchsize,
                        collate_fn=DataCollatorForCellClassification(), shuffle=False)
eval_loader = DataLoader(testdataset.select(range(eval_num)), batch_size=test_batchsize,
                        collate_fn=DataCollatorForCellClassification(), shuffle=False)

# %%
model.train()
text_encoder.train()
loss_fn = nn.CrossEntropyLoss()
optimizer1 = torch.optim.Adam(model.parameters(), lr=1e-5)
optimizer2 = torch.optim.Adam(text_encoder.parameters(), lr=1e-5)

for epoch in range(epochs):
    print('epoch:', epoch)
    for i, d in tqdm(enumerate(train_loader)):
        model.train()
        text_encoder.train()
        text_embs = torch.cat([text_encode(text) for text in texts], 0).T.cuda()
        cell_last_h, cellemb = cell_encode(d['input_ids'], d['attention_mask']) # batchsize * 256
        # text_embs: 256 * class_num
        sim = (cellemb @ text_embs) / 0.05 # batchsize * class_num
        loss_sim = loss_fn(sim, d['labels'].cuda())

        ctm_logit = torch.zeros_like(sim)
        for text_idx, text in enumerate(texts):
            text_list = [text] * sim.shape[0]
            ctm_logit[:, text_idx] = ctm(text_list, cell_last_h, d['attention_mask'])
        loss_ctm = loss_fn(ctm_logit, d['labels'].cuda())

        loss = loss_sim + loss_ctm
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss.backward()
        optimizer1.step()
        optimizer2.step()
    

# %%
cell_embs = torch.zeros(len(testdataset), 256)
model.eval()
text_encoder.eval()
preds = torch.zeros(len(testdataset))
sim_logits = torch.zeros(len(testdataset), text_embs.shape[-1])
ctm_logits = torch.zeros(len(testdataset), text_embs.shape[-1])
logits = torch.zeros(len(testdataset), text_embs.shape[-1])
labels = torch.tensor(testdataset['label'])
text_embs = torch.cat([text_encode(text) for text in texts], 0).T.cuda()
with torch.no_grad():
    for i, d in tqdm(enumerate(test_loader)):
        cell_last_h, cellemb = cell_encode(d['input_ids'], d['attention_mask']) # batchsize * 256
        sim = (cellemb @ text_embs) / 0.05 # batchsize * class_num
        sim_logit = F.softmax(sim, dim=-1)

        # ctm
        ctm_logit = torch.zeros_like(sim_logit)
        for text_idx, text in enumerate(texts):
            text_list = [text] * sim_logit.shape[0]
            ctm_logit[:, text_idx] = ctm(text_list, cell_last_h, d['attention_mask'])
        ctm_logit = F.softmax(ctm_logit, dim=-1)

        sim_logits[i * test_batchsize: (i + 1) * test_batchsize] = sim_logit.cpu()
        ctm_logits[i * test_batchsize: (i + 1) * test_batchsize] = ctm_logit.cpu()
        logit = (sim_logit + ctm_logit) / 2
        pred = logit.argmax(dim=-1)
        logits[i * test_batchsize: (i + 1) * test_batchsize] = logit.cpu()
        cell_embs[i * test_batchsize: (i + 1) * test_batchsize] = cellemb.cpu()
        preds[i * test_batchsize: (i + 1) * test_batchsize] = pred.cpu()

torch.save({'cell_embs': cell_embs, 'text_embs': text_embs, 
            'sim_logits': sim_logits, 'ctm_logits': ctm_logits,
            'preds': preds, 'labels': labels, 'logits': logits}, 
           output_path + 'result.pt')

# %%

from sklearn.metrics import f1_score, accuracy_score

for k in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    preds_k = (k * sim_logits + (1 - k) * ctm_logits).argmax(dim=-1)
    print(k, '\n', accuracy_score(labels, preds_k), f1_score(labels, preds_k, average='macro'))