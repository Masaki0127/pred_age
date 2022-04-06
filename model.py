import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertJapaneseTokenizer, BertModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):

    def __init__(self, d_model=768, dropout=0.2):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(50, d_model)
        position = torch.arange(0, 50, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, time_series):
        x = x + torch.squeeze(self.pe[time_series, :])
        return self.dropout(x)

class Bertmulticlassficationmodel(nn.Module):
    def __init__(self, numlabel, model_name):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert_model = BertModel.from_pretrained(model_name)
        self.pos_encode = PositionalEncoding()
        encoder_layers = TransformerEncoderLayer(d_model=768, nhead=8, dropout=0.2, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 2)
        self.dropout1 = nn.Dropout(p=0.2)
        self.linear = nn.Linear(768, numlabel)

    def forward(self, text, time_series, padding):
        text = torch.reshape(text,(-1, 3, 128))
        x = self.bert_model(input_ids=text[:,0,:], token_type_ids=text[:,1,:], attention_mask=text[:,2,:])[0][:,0,:] #BERTの最終層のCLSを出力
        x = torch.reshape(x,(-1,15,768))
        x = self.pos_encode(x, time_series)
        x = torch.mul(x,torch.unsqueeze(padding,2))
        x = self.transformer_encoder(x, mask=None, src_key_padding_mask=(1-padding).bool())
        x = torch.bmm(torch.unsqueeze(padding, 1),x)
        x = torch.squeeze(x)/torch.unsqueeze(torch.sum(padding,1),1)
        output = self.linear(self.dropout1(x))
        return output

class Algolithm:
    def __init__(self, numlabel, model_name, path = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #GPUがつかえたらGPUを利用
        self.model = nn.DataParallel(Bertmulticlassficationmodel(numlabel, model_name),[0,1,2,3])#モデルをGPUに乗せる
        self.loss_list = []
        self.val_list = []
        result_dir = '/gs/hs0/tga-nakatalab/home/higashi/scaling_nohaba/BERT_trans_LDL_std1_fix' #保存場所
        if not os.path.exists(result_dir): #保存場所が無かったら作成
            os.mkdir(result_dir)

    def train(self, encoder, valid, lr):
        self.model.to(self.device)
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=lr) #optimizer
        criterion = nn.BCEWithLogitsLoss()
        min_score = 10000 #early_stopに使う
        early_stop = True
        ep=1
        while early_stop:
            print(f'epoch:{ep}')
            self.model.train() #モデルを訓練モードに
            running_loss = 0.0
            optimizer.zero_grad()
            j=0
            for data in tqdm(encoder):
                j+=1
                text, age_list, padding, labels = data
                text = text.to(self.device)
                padding = padding.to(self.device)
                
                output = self.model(text, age_list, padding)
                loss = criterion(output, labels.to(self.device))
                loss = loss/8
                loss.backward()
                if j%8==0:
                    optimizer.step()
                    optimizer.zero_grad()
                running_loss += loss.item()

            self.loss_list.append(running_loss/len(encoder))

            #検証 & early_stop
            score = self.loss_exact(valid)
            if min_score > score:
                min_score = score
                s = 0
            else:
                s+=1
            if s==5:
                early_stop = False
            save_path = f"/gs/hs0/tga-nakatalab/home/higashi/scaling_nohaba/BERT_trans_LDL_std1_fix/model{ep}.pth"
            torch.save({"epoch": ep, "model_state_dict": self.model.module.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, save_path)
            print(running_loss/len(encoder))
            ep+=1
        print(f"train_loss:{self.loss_list}")
        print('Finished Training')
        print(f'Saved checkpoint at {save_path}')
        return ep-6
    @torch.no_grad()
    def loss_exact(self, test):
        self.model.eval()
        all = 0
        vali_loss = 0
        epoch_corrects = 0
        criterion = nn.BCEWithLogitsLoss()
        for data in tqdm(test):
            text, age_list, padding, labels = data
            text = text.to(self.device)
            padding = padding.to(self.device)

            output = self.model(text, age_list, padding)
            loss = criterion(output, labels.to(self.device))
            all+=len(data)
            vali_loss += loss.item()
        print(f"vali_loss:{vali_loss/len(test)}")
        return vali_loss/len(test)
