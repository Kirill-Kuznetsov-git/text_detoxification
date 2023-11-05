from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pandas as pd

PROJECT_PATH = Path(__file__).parent.parent.parent.resolve().__str__()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
CHECKPOINT_PATH = PROJECT_PATH + "/models/best"
MODEL_NAME = 't5-base'
TEST_INTERIM_PATH = PROJECT_PATH + "/data/interim/test/tokenized.tsv"


model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT_PATH)
model.eval()
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def detox(model, inference_request, tokenizer=tokenizer):
    input_ids = tokenizer(inference_request, return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True,temperature=0))


def from_serias_to_tensor(array):
    input_ids = [str_input_ids.split(', ') for str_input_ids in array]
    input_ids = [[int(input_id_str[0][1:])] + [int(i) for i in input_id_str[1:-1]] + [int(input_id_str[-1][:-1])] + [0] * (128 - len(input_id_str)) 
                        for input_id_str in input_ids]
    input_ids = torch.tensor(input_ids)

    return input_ids

class TokenizedDataset(Dataset):
    def __init__(self, dataset_path):
        tokenized_df = pd.read_csv(dataset_path, sep='\t')
        
        self.input_ids = from_serias_to_tensor(tokenized_df.input_ids)
        self.attention_mask = from_serias_to_tensor(tokenized_df.attention_mask)
        self.labels = from_serias_to_tensor(tokenized_df.labels)
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return {"input_ids": self.input_ids[index], "attention_mask": self.attention_mask[index], "labels": self.labels[index]}

test_tokenized_dataset = TokenizedDataset(TEST_INTERIM_PATH)

batch_size = 16
args = Seq2SeqTrainingArguments(
    f"{MODEL_NAME}-finetuned-detoxigication",
    evaluation_strategy = "epoch",
    learning_rate=1e-3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=10,
    predict_with_generate=True,
    fp16=True if DEVICE.type == 'cuda' else False,
    report_to='tensorboard',
)

trainer = Seq2SeqTrainer(
    model,
    args,
    eval_dataset=test_tokenized_dataset,
)

trainer.evaluate()
