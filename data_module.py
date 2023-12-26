import pandas as pd
import torch
import pytorch_lightning as pl
import transformers
from tqdm.auto import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import Subset


# Basic Dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, idx):
        if len(self.targets) == 0:
            return {"input_ids": self.inputs[idx]["input_ids"],
                    "attention_mask": self.inputs[idx]["attention_mask"]}
        else:
            return {"input_ids": self.inputs[idx]["input_ids"],
                    "attention_mask": self.inputs[idx]["attention_mask"]}, torch.tensor(self.targets[idx])
        
    def __len__(self):
        return len(self.inputs)
    

# Dataset for Similarity model
class SimilarityDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs

        self.targets = targets

    def __getitem__(self, idx):
        if len(self.targets) == 0:
            return {"input_ids_1": self.inputs[idx]["input_ids_1"],
                    "attention_mask_1": self.inputs[idx]["attention_mask_1"],
                    "input_ids_2": self.inputs[idx]["input_ids_2"],
                    "attention_mask_2": self.inputs[idx]["attention_mask_2"]}
        else:
            return {"input_ids_1": self.inputs[idx]["input_ids_1"],
                    "attention_mask_1": self.inputs[idx]["attention_mask_1"],
                    'input_ids_2': self.inputs[idx]['input_ids_2'],
                    'attention_mask_2': self.inputs[idx]['attention_mask_2']}, torch.tensor(self.targets[idx])

    def __len__(self):
        return len(self.inputs)
    

# Basic Dataloader
class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle


        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=160)

        # special token extracted from 'source' column
        # need to resize token embeddings in the model
        special_tokens_dict = {
            "additional_special_tokens": [
                "[petition]",
                "[nsmc]",
                "[slack]",
                "[sampled]",
                "[rtt]",
            ]
        }

        self.tokenizer.add_special_tokens(special_tokens_dict)

        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # [CLS] [source1] [source2] [sentence1] [SEP] [sentence2] [SEP]
            src_tokens = [f"[{src}]" for src in item['source'].split("-")]
            text = ''.join(src_tokens) + '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=160)
            for key in outputs:
                outputs[key] = torch.tensor(outputs[key], dtype=torch.long)
                
            data.append(outputs)
        return data

    def preprocessing(self, data):
        data = data.drop(columns=self.delete_columns)
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            # 학습데이터 준비
            train_inputs, train_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, _ = self.preprocessing(predict_data) # predict inputs, predict targets
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)
    

# Basic Dataloader for KFold cross validation
class KFoldDataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path, 
                 k: int = 1, split_seed: int = 42, num_splits: int = 5):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.k = k
        self.split_seed = split_seed
        self.num_splits = num_splits


        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=160)

        special_tokens_dict = {
            "additional_special_tokens": [
                "[petition]",
                "[nsmc]",
                "[slack]",
                "[sampled]",
                "[rtt]",
            ]
        }

        self.tokenizer.add_special_tokens(special_tokens_dict)

        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            src_tokens = [f"[{src}]" for src in item['source'].split("-")]
            text = ''.join(src_tokens) + '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            for key in outputs:
                outputs[key] = torch.tensor(outputs[key], dtype=torch.long)

            data.append(outputs)
        return data


    def preprocessing(self, data):
        data = data.drop(columns=self.delete_columns)

        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            total_data = pd.concat([train_data, val_data], axis=0).reset_index(drop=True)
            total_inputs, total_targets = self.preprocessing(total_data)
            total_dataset = Dataset(total_inputs, total_targets)

            # k-fold cross validation을 위한 데이터셋을 준비합니다
            # only binary/multi-class classification supports StratifiedKFold
            kf = KFold(n_splits=self.num_splits, shuffle=self.shuffle, random_state=self.split_seed)
            all_splits = [k for k in kf.split(total_dataset)]
            # k번째 fold에 속하는 data 들의 index를 가져옵니다
            train_idx, val_idx = all_splits[self.k]
            train_idx, val_idx = train_idx.tolist(), val_idx.tolist()

            self.train_dataset = Subset(total_dataset, train_idx)
            self.val_dataset = Subset(total_dataset, val_idx)
        else:
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, _ = self.preprocessing(predict_data) # predict inputs, predict targets
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)
    

# Dataloader for Similarity model
class SimilarityDataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer_1 = transformers.AutoTokenizer.from_pretrained(model_name, max_length=160)
        self.tokenizer_2 = transformers.AutoTokenizer.from_pretrained('jhgan/ko-sroberta-multitask', max_length=160)

        special_tokens_dict = {
            "additional_special_tokens": [
                "[petition]",
                "[nsmc]",
                "[slack]",
                "[sampled]",
                "[rtt]",
            ]
        }

        self.tokenizer_1.add_special_tokens(special_tokens_dict)

        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']
        self.source_columns = ['source']


    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            s1, s2 = item[self.source_columns].item().split('-')

            text=f'[{s1}]'+f'[{s2}]'+text # source token을 추가

            outputs_1 = self.tokenizer_1(text, add_special_tokens=True, padding='max_length', truncation=True)

            # sroberta
            outputs_2 = self.tokenizer_2([item['sentence_1'],item['sentence_2']], padding='max_length', truncation=True, return_tensors='pt')

            total_outputs={}
            total_outputs['input_ids_1']=torch.tensor(outputs_1['input_ids'], dtype=torch.long)
            total_outputs['attention_mask_1']=torch.tensor(outputs_1['attention_mask'], dtype=torch.long)
            total_outputs['input_ids_2']=outputs_2['input_ids']
            total_outputs['attention_mask_2']=outputs_2['attention_mask']

            data.append(total_outputs)
        return data

    def preprocessing(self, data):
        data = data.drop(columns=self.delete_columns)

        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            train_inputs, train_targets = self.preprocessing(train_data)

            val_inputs, val_targets = self.preprocessing(val_data)

            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)