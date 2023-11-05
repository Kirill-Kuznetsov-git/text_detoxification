from transformers import AutoTokenizer, BertForMaskedLM
import torch

MODEL_NAME = 't5-base'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = BertForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)

