import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os
import pickle
import pandas as pd

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import json
from copy import deepcopy
# from torchvision.models import resnet50
from torch.utils.data import DataLoader

from sklearn.metrics import mean_absolute_error
from Attention import self_attention, MultiHeadAttention
from ResNet50 import resnet50

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


print('data preparing...')

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


# load pickle file
with open("./Data/gait_data.pickle","rb") as fr:
    data = pickle.load(fr)


# load train, validate
with open("./Data/train_keys.pkl","rb") as fr:
    train_keys = pickle.load(fr)
with open("./Data/valid_keys.pkl","rb") as fr:
    valid_keys = pickle.load(fr)
with open("./Data/test_keys.pkl","rb") as fr:
    test_keys = pickle.load(fr)

## DataLoader
class OUISIRgait(Dataset):
    def __init__(self, keys, data):
        # Mean and Std for ImageNet
        mean=[0.485, 0.456, 0.406] # ImageNet
        std=[0.229, 0.224, 0.225] # ImageNet


        self.images = []
        self.gait_feat = []
        self.ages = []
        self.genders = []
        self.subjects = []
        self.pose_vec = []
        
        # Set Inputs and Labels
        for i in keys:
            self.images.append(data[i][1])
            self.genders.append(data[i][2])
            self.ages.append(data[i][3])
            self.subjects.append(data[i][4])
            self.gait_feat.append(np.array(data[i][5]))
            self.pose_vec.append(np.array(data[i][6][:18]))
    
    def __len__(self):
         return len(self.images)

    def __getitem__(self, index):
        # Load an Image
        img = Image.open(self.images[index]).convert('RGB')
        img = T.ToTensor()(T.Resize((224,224))(img))

        # Get the Labels
        age = self.ages[index]
        gender = self.genders[index]
        subject = self.subjects[index]
        gait_feat = torch.from_numpy(self.gait_feat[index]).float()
        pose_vec = torch.from_numpy(self.pose_vec[index]).float()
        
        # Return the sample of the dataset
        sample = {'image':img, 'age': age, 'gender': gender, 'subject': subject, 'gait_feat': gait_feat, 'pose_vec': pose_vec}
        return sample


BATCH_SIZE = 32

train_dataloader = DataLoader(OUISIRgait(train_keys, data), shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
val_dataloader = DataLoader(OUISIRgait(valid_keys, data), shuffle=False, batch_size=BATCH_SIZE, drop_last=True)


train_steps = len(train_dataloader.dataset) // BATCH_SIZE
val_steps = len(val_dataloader.dataset) // BATCH_SIZE


print('data compelete!')


## Model
with open("./config.json") as json_data_file:
    data = json.load(json_data_file)

is_it_bidirectional = data["is_it_bidirectional"] #True
multi_modal = data["multi_modal"] #True
shall_i_have_contextual = data["shall_i_have_contextual"] #True
shall_i_have_inter_segment = data["shall_i_have_inter_segment"] #True

encoder_shape = 128
HIDDEN_DIM =  128
START_AGE = 2
END_AGE = 87


class GaitMix(nn.Module):
    """
    Main classifier
    """  
    def __init__(self, input_size, hidden_size, num_layers, att_size):
        super(GaitMix, self).__init__()

        global is_it_bidirectional, multi_modal, shall_i_have_contextual, shall_i_have_contextual, projection_shape

        self.multi_modal = multi_modal
        self.shall_i_have_inter_segment = shall_i_have_inter_segment
        self.contextual = shall_i_have_contextual
        self.shall_i_have_contextual = shall_i_have_contextual

        self.is_it_bidirectional_numerical = is_it_bidirectional * 1
        l = num_layers * (self.is_it_bidirectional_numerical + 1)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.resnet_final_size = 256
        self.att_size = att_size
        self.encoder_shape = encoder_shape
        
        self.gender_targ_size = 1
        self.age_targ_size = END_AGE - START_AGE + 1

        self.multi_model_size = 128
        
        self.mega_cat_size = 5 * self.multi_model_size


        # Linear layer - Pre-shared LSTM
        self.linear_preshare_M = nn.Linear(input_size, self.resnet_final_size)

        # Shared RNN - 2 modalities
        self.rnn_S = nn.LSTM(self.resnet_final_size, hidden_size, num_layers, bidirectional = is_it_bidirectional, batch_first = True)

        # RNN for Pose vectors embeddings
        self.rnn_P = nn.LSTM(72, hidden_size, num_layers, bidirectional = is_it_bidirectional, batch_first=True)

        # RNN for Gait features embeddings
        self.rnn_G = nn.LSTM(11, hidden_size, num_layers, bidirectional = is_it_bidirectional, batch_first=True)
        
        # RNN for Images' ResNet embedding
        self.rnn_I = nn.LSTM(self.resnet_final_size, hidden_size, num_layers, bidirectional = is_it_bidirectional, batch_first=True)

        # ResNet 50
        self.resnet = resnet50()
        
        # Self Attention - Shared
        self.SelfAtten_shared = self_attention(l*hidden_size, self.att_size, self.is_it_bidirectional_numerical + 1)        

        # Self Attention - Pose vectors
        self.SelfAtten_pose = self_attention(l*hidden_size, self.att_size, self.is_it_bidirectional_numerical + 1)

        # Self Attention - Gait characteristic
        self.SelfAtten_gaitfeat = self_attention(l*hidden_size, self.att_size, self.is_it_bidirectional_numerical + 1)
        
        # Self Attention for image embedding
        self.SelfAtten_img = self_attention(l*hidden_size, self.att_size, self.is_it_bidirectional_numerical + 1)

        # Linear layer - Shared
        self.linear_shared = nn.Linear(att_size, self.multi_model_size)
        
        # Linear layer - Image 64,64
        self.linear_img = nn.Linear(att_size, self.multi_model_size)
        
        # Linear Layer - mesh concat
        self.linear_meshconcat = nn.Linear(128, self.multi_model_size)
        
        # Cross Attention Layer
        self.cross_attn = MultiHeadAttention()

        # Mega Linear Layers 640 -> 512
        self.linear1_mega = nn.Linear(self.mega_cat_size, 512)
        self.linear2_mega = nn.Linear(512, 256)
        self.linear3_mega = nn.Linear(256, self.encoder_shape)

        self.final_linear_gender = nn.Linear(self.encoder_shape, self.gender_targ_size)
        self.final_linear_age = nn.Linear(self.encoder_shape, self.age_targ_size)

    def forward(self, pose_vec, img, gait_feat):

        # global resnet18
        R = torch.nn.ReLU()
        D = torch.nn.Dropout(0.3)
        N = gait_feat.shape[0] 
        
        ## GEI resnet, global average pooling
        resnet_output = self.resnet(img).view(N, 8, self.resnet_final_size)
        
        pose_concat1 = torch.cat((pose_vec, gait_feat), dim = -1) # 18,83
        
        shared_pose = D(R(self.linear_preshare_M(pose_concat1))) # 18,256
        
        ## shared bilstm layer를 위한 관절벡터(mesh), 이미지(GEI) 결합
        Concat_shared = torch.cat((shared_pose, resnet_output), dim = 1)
        
        ## Mesh 메모리 삭제
        del shared_pose
        
        # fusion module
        S_encoder, (hidden, cell) = self.rnn_S(Concat_shared)
        hidden = R(hidden.permute(1, 0, 2).flatten(-2)) 
        S_attn = self.SelfAtten_shared(S_encoder, hidden)
        shared1 = D(R(self.linear_shared(S_attn)))

        # 3D pose module
        P_encoder, (hidden, cell) = self.rnn_P(pose_vec)
        hidden = R(hidden.permute(1, 0, 2).flatten(-2))
        P_atten = self.SelfAtten_pose(P_encoder, hidden) #32,64
        
        G_encoder, (hidden, cell) = self.rnn_G(gait_feat)
        hidden = R(hidden.permute(1, 0, 2).flatten(-2))
        G_atten = self.SelfAtten_gaitfeat(G_encoder, hidden) #32,64
        
        pose1 = torch.cat((P_atten, G_atten), dim = -1) # 32,128
        
        pose_concat2 = D(R(self.linear_meshconcat(pose1))) #32,128
        
        del pose_vec, img, gait_feat

        # silhouette module
        I_encoder, (hidden, cell) = self.rnn_I(resnet_output)
        hidden = R(hidden.permute(1, 0, 2).flatten(-2))
        I_atten = self.SelfAtten_img(I_encoder, hidden)
        img1 = D(R(self.linear_img(I_atten)))
        
        del P_encoder, I_encoder, hidden, G_encoder

        if self.multi_modal:
            
            # 32,128
            CrossAtt_GM = self.cross_attn(img1, pose_concat2, pose_concat2)[0].squeeze()
            CrossAtt_MG = self.cross_attn(pose_concat2, img1, img1)[0].squeeze()
            
            # 32,640
            mega_concat = R(torch.cat((pose_concat2, img1, shared1, CrossAtt_GM, CrossAtt_MG), dim = -1))
        
            multi_modal_final = D(R(self.linear3_mega(D(R(self.linear2_mega(D(R((self.linear1_mega(mega_concat))))))))))
            
        
        gender = self.final_linear_gender(multi_modal_final)
        age = self.final_linear_age(multi_modal_final)

        return gender, age


with torch.no_grad():
    torch.cuda.empty_cache()
    
with open("config.json") as json_data_file:
    data = json.load(json_data_file)

print('Loading the config.json file ...')

epochs = data[ "epochs"]
lr = data[ "lr"]
lstm_hidden_size = data["lstm_hidden_size"]
file_name = data["file_name"]
num_layers = data['num_layers']
att_size = data['att_size'] # 64
batch_size = data['batch_size']

input_size = 83

centralized_model = GaitMix(input_size, lstm_hidden_size, num_layers, att_size).to(device=device)
    
gender_loss = torch.nn.BCELoss() 

criterion1 = torch.nn.CrossEntropyLoss().to(device=device)
criterion2 = torch.nn.L1Loss().to(device=device)

optimizer = optim.Adam(centralized_model.parameters(), lr = lr)


print('Training ...')

BEST_VALID_LOSS = 1e+10

logger = {"train_loss": list(),
          "validation_loss": list(),
          "train_gender_loss": list(),
          "train_age_loss": list(),
          "validation_gender_loss": list(),
          "validation_age_loss": list(),
          }


sig = nn.Sigmoid()


for e in range(epochs):
    
    total_training_loss = 0
    training_gender_loss = 0
    training_age_loss = 0

    training_gender_correct = 0
    training_age_predicted = list()
    training_age_groudtruth = list()
    
    centralized_model.train()
    
    for i, data in enumerate(train_dataloader):
        
        optimizer.zero_grad()
        
        x1 = data["pose_vec"].to(device=device)
        x2 = data["image"].to(device=device)
        x3 = data["gait_feat"].to(device=device)
        
        gender_output, age_output = centralized_model(x1, x2, x3)
         
        gender_label = data["gender"].to(device=device)
        age_label = data["age"].to(device=device)
        
        loss_1 = gender_loss(sig(gender_output), gender_label.unsqueeze(1).float())
        
        softmax_loss = criterion1(age_output, age_label-2)
    
        m = nn.Softmax(dim=1)
        output_softmax = m(age_output)
        a = torch.arange(START_AGE, END_AGE + 1, dtype=torch.float32).to(device=device)
        age_pred = (output_softmax * a).sum(1, keepdim=True).cpu().data.numpy()

        loss_l1 = criterion2(torch.Tensor(age_pred).to(device=device), age_label.unsqueeze(1).float())
        
        loss_2 = softmax_loss + loss_l1
        
        loss = loss_1 + (0.7*loss_2)

        loss.backward()
        optimizer.step()
        total_training_loss += loss
        
        training_gender_loss += loss_1.item()
        training_age_loss += loss_2.item()

        gender_predicted = list()
    
        for i in sig(gender_output):
            if i >=0.5:
                gender_predicted.append(1)
            else:
                gender_predicted.append(0)    
           
        for i in range(len(gender_label)):
            if gender_predicted[i] == gender_label[i]:
                training_gender_correct += 1

        training_age_predicted.extend(np.ravel(age_pred, order='C'))
        training_age_groudtruth.extend(age_label.tolist())

    avgTrainGenderLoss = training_gender_loss/len(train_dataloader.dataset)
    avgTrainAgeLoss = training_age_loss/len(train_dataloader.dataset)
    
    ## age MAE
    mae = mean_absolute_error(training_age_groudtruth, training_age_predicted)
        
    ## gender accuracy            
    ccr = training_gender_correct/len(training_age_groudtruth)
    
    print('EPOCH ', e+1)
    print("Training Losses: Gender: {}, Age: {}".format(avgTrainGenderLoss, avgTrainAgeLoss))
    print("Training gender CCR: {}, Age MAE: {}".format(ccr, mae))
    print('----------------')
    
    with torch.no_grad():
        total_validation_loss = 0
        validation_gender_loss = 0
        validation_age_loss = 0
       
        validation_gender_correct = 0
        validation_age_predicted = list()
        validation_age_groudtruth = list()
        
        centralized_model.eval()
        gender_correct = 0
        
        for i, data in enumerate(val_dataloader):
            
            x1 = data["pose_vec"].to(device=device)
            x2 = data["image"].to(device=device)
            x3 = data["gait_feat"].to(device=device)

            gender_output, age_output = centralized_model(x1, x2, x3)

            gender_label = data["gender"].to(device=device)
            age_label = data["age"].to(device=device)
            
            loss_1 = gender_loss(sig(gender_output), gender_label.unsqueeze(1).float())

            softmax_loss = criterion1(age_output, age_label-2)
    
            m = nn.Softmax(dim=1)
            output_softmax = m(age_output)
            a = torch.arange(START_AGE, END_AGE + 1, dtype=torch.float32).to(device=device)
            age_pred = (output_softmax * a).sum(1, keepdim=True).cpu().data.numpy()

            loss_l1 = criterion2(torch.Tensor(age_pred).to(device=device), age_label.unsqueeze(1).float())
            
            loss_2 = softmax_loss + loss_l1
        
            loss = loss_1 + (0.7*loss_2)
            
            total_validation_loss += loss

            validation_gender_loss += loss_1.item()
            validation_age_loss += loss_2.item()

            gender_predicted = list()
            for i in sig(gender_output):
                if i >=0.5:
                    gender_predicted.append(1)
                else:
                    gender_predicted.append(0)
                    
            for i in range(len(gender_label)):
                if gender_predicted[i] == gender_label[i]:
                    validation_gender_correct += 1
            
            validation_age_predicted.extend(np.ravel(age_pred, order='C'))
            validation_age_groudtruth.extend(age_label.tolist())
            
        avgValGenderLoss = validation_gender_loss/len(val_dataloader.dataset)
        avgValAgeLoss = validation_age_loss/len(val_dataloader.dataset)    
        
        ## age MAE
        mae = mean_absolute_error(validation_age_groudtruth, validation_age_predicted)
        
        ## gender accuracy            
        ccr = validation_gender_correct/len(validation_age_groudtruth)
        
        print("Validation Losses: Gender: {}, Age: {}".format(avgValGenderLoss, avgValAgeLoss))
        print("Validation gender CCR: {}, Age MAE: {}".format(ccr, mae))
        print('----------------')
        
    avgTrainLoss = total_training_loss / train_steps
    avgValLoss = total_validation_loss / val_steps
    
    print('Average Losses — Training: {} | Validation {}'.format(avgTrainLoss, avgValLoss))
    print()
    
    avgTrainGenderLoss = training_gender_loss/len(train_dataloader.dataset)
    avgTrainAgeLoss = training_age_loss/len(train_dataloader.dataset)

    avgValGenderLoss = validation_gender_loss/len(val_dataloader.dataset)
    avgValAgeLoss = validation_age_loss/len(val_dataloader.dataset)

    logger["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    logger["train_gender_loss"].append(avgTrainGenderLoss)
    logger["train_age_loss"].append(avgTrainAgeLoss)
    
    logger["validation_loss"].append(avgValLoss.cpu().detach().numpy())
    logger["validation_gender_loss"].append(avgValGenderLoss)
    logger["validation_age_loss"].append(avgValAgeLoss)
    
    if avgValLoss < BEST_VALID_LOSS :
        torch.save(centralized_model.state_dict(), './Results/model_state_dict.pt')
        BEST_VALID_LOSS = avgValLoss
    if os.path.isfile('./Results/valid_result.txt'):
        with open('./Results/valid_result.txt', 'a') as f:
            f.write("Epoch: {}, Validation gender accuracy: {}, Age MAE: {}".format(e ,ccr, mae) + '\n')
    else:
        with open('./Results/valid_result.txt', 'w') as f:
            f.write("Epoch: {}, Validation gender accuracy: {}, Age MAE: {}".format(e, ccr, mae) + '\n')

print('Training compelete!')

