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
from torchvision.models import resnet50
from torch.utils.data import DataLoader

from sklearn.metrics import mean_absolute_error
from Attention import self_attention, MultiHeadAttention

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
with open("./gait_data.pickle","rb") as fr:
    data = pickle.load(fr)

# load test
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

test_dataloader = DataLoader(OUISIRgait(test_keys, data), shuffle=False, batch_size=BATCH_SIZE, drop_last=True)
test_steps = len(test_dataloader.dataset) // BATCH_SIZE

print('data compelete!')

## Model
with open("./config.json") as json_data_file:
    data = json.load(json_data_file)

is_it_bidirectional = data["is_it_bidirectional"] #True
multi_modal = data["multi_modal"] #True
shall_i_have_contextual = data["shall_i_have_contextual"] #True
shall_i_have_inter_segment = data["shall_i_have_inter_segment"] #True

encoder_shape = 128

    
resnet50 = models.resnet50(pretrained=True)
resnet50 = torch.nn.Sequential(*(list(resnet50.children())[:-3])).to(device)

for param in resnet50.parameters():
    param.requires_grad = False

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

HIDDEN_DIM =  128
START_AGE = 2
END_AGE = 87

class ASPFuseNet(nn.Module):
    """
    Main classifier
    """  
    def __init__(self, input_size, hidden_size, num_layers, att_size):
        super(ASPFuseNet, self).__init__()

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

    ## 여기서 r은 review, t는 text
    def forward(self, pose_vec, img, gait_feat):

        # global resnet18
        R = torch.nn.ReLU()
        D = torch.nn.Dropout(0.2)
        N = gait_feat.shape[0] 
        
        ## GEI resnet, global average pooling
        resnet_glob_avg_pool = resnet50(img).mean(dim=(-2, -1)).unsqueeze(-1).view(N, 4, self.resnet_final_size) # 4,256
        
        pose_concat1 = torch.cat((pose_vec, gait_feat), dim = -1) # 18,83
        
        shared_pose = D(R(self.linear_preshare_M(pose_concat1))) # 18,256
        
        ## shared bilstm layer를 위한 관절벡터(mesh), 이미지(GEI) 결합
        Concat_shared = torch.cat((shared_pose, resnet_glob_avg_pool), dim = 1)
        
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
        I_encoder, (hidden, cell) = self.rnn_I(resnet_glob_avg_pool)
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


model = ASPFuseNet(input_size, lstm_hidden_size, num_layers, att_size).to(device)
model.load_state_dict(torch.load("../Results/model_state_dict.pt", map_location=device))

with torch.no_grad():
    sig = nn.Sigmoid()
    
    gender_correct = 0
    count = 0
    
    age_predicted = list()
    age_groudtruth = list()
    
    model.eval()
    
    for i, data in enumerate(test_dataloader):
        
        x1 = data["pose_vec"].to(device=device)
        x2 = data["image"].to(device=device)
        x3 = data["gait_feat"].to(device=device)

        gender_output, age_output = model(x1, x2, x3)
#         gender_output = model(x1, x2, x3)
        
        gender_label = data["gender"].to(device='cpu')
        age_label = data["age"].to(device='cpu')
        
        gender_predicted = list()
        
        for i in sig(gender_output):
            if i >=0.5:
                gender_predicted.append(1)
            else:
                gender_predicted.append(0)
            
        count += len(gender_predicted)
           
        for i in range(len(gender_label)):
            if gender_predicted[i] == gender_label[i]:
                gender_correct += 1
                
        m = nn.Softmax(dim=1)
        output_softmax = m(age_output)
        a = torch.arange(START_AGE, END_AGE + 1, dtype=torch.float32).to(device=device)
        age_pred = (output_softmax * a).sum(1, keepdim=True).cpu().data.numpy()
        
#         age_predicted.extend(age_output.tolist())
        age_predicted.extend(np.ravel(age_pred, order='C'))
        age_groudtruth.extend(age_label.tolist())
                
    # age MAE
    mae = mean_absolute_error(age_groudtruth, age_predicted)
    
    # age CS
    cs1_sum = 0
    cs5_sum = 0
    cs10_sum = 0
    
    age_predicted = np.ravel(age_predicted, order='C')

    for n in range(len(age_predicted)):
        if abs(age_groudtruth[n]-age_predicted[n]) <= 1:
            cs1_sum += 1
            cs5_sum += 1
            cs10_sum += 1
        elif abs(age_groudtruth[n]-age_predicted[n]) <= 5:
            cs5_sum += 1
            cs10_sum += 1
        elif abs(age_groudtruth[n]-age_predicted[n]) <= 10:
            cs10_sum += 1

    cs1 = cs1_sum/len(age_predicted)
    cs5 = cs5_sum/len(age_predicted)
    cs10 = cs10_sum/len(age_predicted)
        
    ## gender accuracy            
    accuracy = gender_correct/count
        
    print("Test gender accuracy: {}, Age MAE: {}, CS(1): {}, CS(5): {}, CS(10): {}".format(accuracy, mae, cs1, cs5, cs10))