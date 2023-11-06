import pandas as pd
from tqdm import tqdm
import pickle

info = pd.read_csv('./Data/subject_info_OUMVLP.csv')

nan = info[info['age'] == '-'].index
info2 = info.drop(nan)

age_group = list()

for i in range(len(info2)):
    age = int(info2.iloc[i].age)
    if age >= 0 and age <= 5:
        age_group.append('0-5')
    elif age >= 6 and age <= 10:
        age_group.append('6-10')
    elif age >= 11 and age <= 15:
        age_group.append('11-15')
    elif age >= 16 and age <= 20:
        age_group.append('16-20')
    elif age >= 21 and age <= 25:
        age_group.append('21-25')
    elif age >= 26 and age <= 30:
        age_group.append('26-30')
    elif age >= 31 and age <= 35:
        age_group.append('31-35')
    elif age >= 36 and age <= 40:
        age_group.append('36-40')
    elif age >= 41 and age <= 45:
        age_group.append('41-45')
    elif age >= 46 and age <= 50:
        age_group.append('46-50')
    elif age >= 51 and age <= 55:
        age_group.append('51-55')
    elif age >= 56 and age <= 60:
        age_group.append('56-60')
    elif age >= 61 and age <= 65:
        age_group.append('61-65')
    elif age >= 66 and age <= 70:
        age_group.append('66-70')
    elif age >= 71 and age <= 75:
        age_group.append('71-75')
    elif age >= 76 and age <= 80:
        age_group.append('56-60')
    elif age >= 81 and age <= 85:
        age_group.append('61-65')
    elif age >= 86 and age <= 90:
        age_group.append('86-90')
    elif age >= 91 and age <= 95:
        age_group.append('91-95')
    elif age >= 96 and age <= 100:
        age_group.append('96-100')
        
info2['age_group'] = age_group

# firstly, we split the data with 8:2 ratio to make train, test set ids.
train_id = list()
test_id = list()

for i in info2.age_group.unique():
    train_temp = list()
    F_temp = info2[(info2['age_group']== i) & (info2['gender']=='F')].ID.tolist()
    test_temp = random.sample(F_temp, int(len(F_temp)*0.2))
    for j in F_temp:
        if j not in test_temp:
            train_temp.append(j)
    train_id.extend(train_temp)
    test_id.extend(test_temp)
    
for i in info2.age_group.unique():
    train_temp = list()
    M_temp = info2[(info2['age_group']== i) & (info2['gender']=='M')].ID.tolist()
    test_temp = random.sample(M_temp, int(len(M_temp)*0.2))
    for j in M_temp:
        if j not in test_temp:
            train_temp.append(j)
    train_id.extend(train_temp)
    test_id.extend(test_temp)

# Secondly, we split train set with the 8:2 ratio to make validation set. 
train_info = pd.DataFrame(columns = ["ID","gender","age", "age_group"])
n = 0

for i in train_id:
    temp_df = info2[info2['ID'] == int(i)]
    train_info.loc[n] = temp_df.values[0]
    n += 1

train2_id = list()
validation_id = list()

for i in train_info.age_group.unique():
    train_temp = list()
    F_temp = train_info[(train_info['age_group']== i) & (train_info['gender']=='F')].ID.tolist()
    val_temp = random.sample(F_temp, int(len(F_temp)*0.125))
    for j in F_temp:
        if j not in val_temp:
            train_temp.append(j)
    train2_id.extend(train_temp)
    validation_id.extend(val_temp)
    
for i in train_info.age_group.unique():
    train_temp = list()
    M_temp = train_info[(train_info['age_group']== i) & (train_info['gender']=='M')].ID.tolist()
    val_temp = random.sample(M_temp, int(len(M_temp)*0.125))
    for j in M_temp:
        if j not in val_temp:
            train_temp.append(j)
    train2_id.extend(train_temp)
    validation_id.extend(val_temp)


print('No of train samples', len(train2_id))
print('No of validation samples', len(validation_id))
print('No of test Samples', len(test_id))

train_keys = list()
valid_keys = list()
test_keys = list()

for i in data.keys():
    if int(i.split('/')[0]) in train2_id:
        train_keys.append(i)
    elif int(i.split('/')[0]) in validation_id:
        valid_keys.append(i)
    elif int(i.split('/')[0]) in test_id:
        test_keys.append(i)
        
with open("./Data/train_keys.pkl","wb") as f:
    pickle.dump(train_keys, f)
with open("./Data/valid_keys.pkl","wb") as f:
    pickle.dump(valid_keys, f)
with open("./Data/test_keys.pkl","wb") as f:
    pickle.dump(test_keys, f)

