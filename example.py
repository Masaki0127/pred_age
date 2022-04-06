df_1 = pd.read_pickle("/gs/hs0/tga-nakatalab/home/higashi/question_cat_1_3_month_label.pkl")
text_list = df_1["content"].values.tolist()
label_list = df_1["child_age"].values.tolist()
age_list = df_1["created"].values.tolist()
flat_label =  [x for row in label_list for x in row]
min_age = min(flat_label)
for i in range(len(label_list)):
    label_list[i]=np.array(label_list[i])-min_age #ラベルの最小値を0にする(onehotにするために)

for i in tqdm(range(len(label_list))): #-9~23は月ごとに24~35で1つのグループ、36~47で1つのグループ、48~59で1つのグループ、60~71で1つのグループ、72~で1つのグループにする
    for j in range(len(label_list[i])):
        if label_list[i][j]>=81:
            label_list[i][j]=81
age_time_list=[]
for i in tqdm(age_list):
    age_time_list.append(age_limit(i,15))

padding = torch.zeros((159265,15))
for i in range(len(text_list)):
    padding[i] = make_padding(text_list[i],15)

for i in tqdm(range(len(text_list))):
    text_list[i]=make_limit(text_list[i],15)