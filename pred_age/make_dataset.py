from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from pred_age import label_tr

class createDataset(Dataset):
    def __init__(self, X, y, z, w, max_length, tokenizer):
        self.X = X
        self.y = y
        self.z = z
        self.w = w
        self.max_length = max_length
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        text = self.X[index]
        text_token = to_token(text, self.max_length, self.tokenizer)
        created_time = self.y[index]
        padding = self.z[index]
        labels = self.w[index]
        return text_token, created_time, padding, labels

def split_data(text_list, created_list, padding, label_list, test_size=0.2, valid_size=0.2, valid=True, random_state = None):
    text_train_all, text_test, created_train_all, created_test, padding_train_all, padding_test, label_train_all, label_test = train_test_split(text_list, created_list, padding, label_list, test_size=test_size, random_state = random_state)
    if valid:
        text_train, text_vali, created_train, created_vali, padding_train, padding_vali, label_train, label_vali = train_test_split(text_train_all, created_train_all, padding_train_all, label_train_all, test_size=valid_size/(1-test_size), random_state = random_state)
        return [text_train, created_train, padding_train, label_train], [text_vali, created_vali, padding_vali, label_vali], [text_test, created_test, padding_test, label_test]

    else:
        text_train, created_train, padding_train, label_train = text_train_all, created_train_all, padding_train_all, label_train_all
        return [text_train, created_train, padding_train, label_train], [text_test, created_test, padding_test, label_test]
        

def to_token(x,max_length, tokenizer):
    x = tokenizer(x, max_length=max_length ,padding="max_length", truncation=True, return_tensors="pt")
    x_list=[x['input_ids'],x['token_type_ids'],x['attention_mask']]
    return torch.stack(x_list, dim=1)

def make_dataloader(text, created, padding, label, size, batch_size, tokenizer, max_length = 512, shuffle=True, mldl=True, std=1, scaling=True):
    if mldl:
        label = label_tr.MLDL(label, size, std=std,scaling=scaling)
    else:
        label = label_tr.one_hot(label, size)
    dataset = createDataset(text, created, padding, label, max_length, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle = shuffle)
    return dataloader