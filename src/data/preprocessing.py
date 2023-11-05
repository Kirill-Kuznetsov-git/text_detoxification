from transformers import T5Tokenizer
import pandas as pd
from pathlib import Path

PROJECT_PATH = Path(__file__).parent.parent.parent.resolve().__str__()
ROW_DATA_PATH = PROJECT_PATH + "/data/raw/filtered.tsv"
TRAIN_INTERIM_PATH = PROJECT_PATH + "/data/interim/train/tokenized.tsv"
TEST_INTERIM_PATH = PROJECT_PATH + "/data/interim/test/tokenized.tsv"
PREFIX = "Make sentence less toxic: "
MODEL_NAME = 't5-base'
RANDOM_STATE = 42

MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128

MIN_REFERENCE_TOX = 0.9
MAX_TRANSLATION_TOX = 0.1

# Read tsv file and take only needed columns.
raw_data = pd.read_csv(ROW_DATA_PATH, sep='\t')
raw_data = raw_data[['reference', 'translation', 'ref_tox', 'trn_tox']]
raw_data['reference'] = PREFIX + raw_data['reference']

# Divide data to train and test.
data_train = raw_data.sample(frac=0.95, random_state=RANDOM_STATE)
data_test=raw_data.drop(data_train.index)

# As said in README, for training the model it is best to have sample with high toxicity level
# and its paraphrazed version with low toxicity level. So for train I save references which
# has more then MIN_REFERENCE_TOX tox and translation which has more then MAX_TRANSLATION_TOX tox.
data_train = data_train[data_train['ref_tox'] > MIN_REFERENCE_TOX]
data_train = data_train[data_train['trn_tox'] < MAX_TRANSLATION_TOX]

# Download pretrained tokenizer.
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)

# Tokenize train data and save as tsv file.
train_model_inputs = tokenizer(list(data_train['reference']), max_length=MAX_INPUT_LENGTH, truncation=True)
train_labels = tokenizer(list(data_train['translation']), max_length=MAX_TARGET_LENGTH, truncation=True)
train_model_inputs['labels'] = train_labels['input_ids']
train_model_inputs_df = pd.DataFrame()
train_model_inputs_df['input_ids'] = train_model_inputs['input_ids']
train_model_inputs_df['attention_mask'] = train_model_inputs['attention_mask']
train_model_inputs_df['labels'] = train_model_inputs['labels']
train_model_inputs_df.to_csv(TRAIN_INTERIM_PATH, sep='\t')

# Tokenize test data and save as tsv file.
test_model_inputs = tokenizer(list(data_test['reference']), max_length=MAX_INPUT_LENGTH, truncation=True)
test_labels = tokenizer(list(data_test['translation']), max_length=MAX_TARGET_LENGTH, truncation=True)
test_model_inputs['labels'] = test_labels['input_ids']
test_model_inputs_df = pd.DataFrame()
test_model_inputs_df['input_ids'] = test_model_inputs['input_ids']
test_model_inputs_df['attention_mask'] = test_model_inputs['attention_mask']
test_model_inputs_df['labels'] = test_model_inputs['labels']
test_model_inputs_df.to_csv(TEST_INTERIM_PATH, sep='\t')
