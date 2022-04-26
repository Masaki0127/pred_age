from pred_age import data_make, model, make_dataset, evaluate
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

print("start")
os.chdir("data")
df_1 = pd.read_pickle("data.pkl")
text_list = df_1["content"].values.tolist()[:50]
label_list = df_1["child_age"].values.tolist()[:50]
created_list = df_1["created"].values.tolist()[:50]
print("finish load")

flat_label =  [x for row in label_list for x in row]
min_age = min(flat_label)
for i in range(len(label_list)):
    label_list[i]=np.array(label_list[i])-min_age #ラベルの最小値を0にする(onehotにするために)

for i in tqdm(range(len(label_list))): 
    for j in range(len(label_list[i])):
        if label_list[i][j]>=81:
            label_list[i][j]=81

flat_label =  [x for row in label_list for x in row]
max_age = max(flat_label)+1

padding = data_make.to_padding(text_list, created_list, period=2)
text_list, created_list, pad = padding.pad_data()
train_set ,vali_set, test_set = make_dataset.split_data(text_list, created_list, pad, label_list, random_state=1)
trainloader = make_dataset.make_dataloader(*train_set, size=max_age, batch_size=4, max_length=64)
valiloader = make_dataset.make_dataloader(*vali_set, size=max_age, batch_size=8, max_length=64)
valiloader_2 = make_dataset.make_dataloader(*vali_set, size=max_age, batch_size=8, max_length=64, mldl = False)
testloader = make_dataset.make_dataloader(*test_set, size=max_age, batch_size=8, max_length=64, mldl = False)
print("finish make dataloader")

result_path = "/gs/hs0/tga-nakatalab/home/higashi/pred_age/result"
algorithm = model.Algorithm(numlabel=max_age, result_path = result_path, max_length = 64, period = 2)
ep=algorithm.train(testloader, valiloader, lr = 2e-5, grad_accum_step=2, early_stop_step=1)
model_path = result_path+f"/model{ep}.pth"

algorithm = model.Algorithm(numlabel=max_age, model_path = model_path, max_length = 64, period = 2)
pred_test, label_test = algorithm.predict(testloader)
pred_vali, label_vali = algorithm.predict(valiloader_2)

thresh = evaluate.make_thresh(pred_vali, label_vali)
print(evaluate.evaluate(pred_test, label_test, thresh))