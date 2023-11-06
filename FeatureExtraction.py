import pandas as pd
from tqdm import tqdm
import math
import pickle
import numpy as np
import os

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

# make a pair dataset for GEI and 3D pose
gait2view = os.listdir('./Data/GEI/')
pose2sub = os.listdir('./Data/meshes/joints_thetas/')

data = dict()

for a in range(len(gait2view)):
    li = os.listdir('./Data/GEI/' + gait2view[a])
    pose_view = gait2view[a].replace('-','_')
    for i in li:
        id_ = int(i.split('.png')[0])
        if id_ in info2.ID.values:
            if len(str(id_)) == 5:
                
                pose_path = glob.glob('./Data/meshes/joints_thetas/'+str(id_)+ '/' + pose_view +'/*.json')
                image_path = './Data/GEI/'+gait2view[a]+'/'+str(id_)+'.png'
                gender = info2[info2.ID == id_].gender.values[0]
                age = info2[info2.ID == id_].age.values[0]
                
                data[str(id_) + '/' + pose_view] = [pose_path, image_path, gender, age, int(id_)]

            elif len(str(id_)) ==4:
                pose_path = glob.glob('./Data/meshes/joints_thetas/'+ '0' + str(id_)+ '/' + pose_view +'/*.json')
                image_path = './Data/GEI/'+gait2view[a]+'/'+'0' + str(id_)+'.png'
                gender = info2[info2.ID == id_].gender.values[0]
                age = info2[info2.ID == id_].age.values[0]
                
                data['0'+str(id_) + '/' + pose_view] = [pose_path, image_path, gender, age, int(id_)]

            elif len(str(id_)) ==3:
                pose_path = glob.glob('./Data/meshes/joints_thetas/'+ '00' +str(id_)+ '/' + pose_view +'/*.json')
                image_path = './Data/GEI/'+gait2view[a]+'/'+'00' + str(id_)+'.png'
                gender = info2[info2.ID == id_].gender.values[0]
                age = info2[info2.ID == id_].age.values[0]
                
                data['00'+str(id_) + '/' + pose_view] = [pose_path, image_path, gender, age, int(id_)]

            elif len(str(id_)) ==2:
                pose_path = glob.glob('./Data/meshes/joints_thetas/'+ '000' +str(id_)+ '/' + pose_view +'/*.json')
                image_path = './Data/GEI/'+gait2view[a]+'/'+'000' + str(id_)+'.png'
                gender = info2[info2.ID == id_].gender.values[0]
                age = info2[info2.ID == id_].age.values[0]
                
                data['000'+str(id_) + '/' + pose_view] = [pose_path, image_path, gender, age, int(id_)]

            else:
                pose_path = glob.glob('./Data/meshes/joints_thetas/'+ '0000' +str(id_)+ '/' + pose_view +'/*.json')
                image_path = './Data/GEI/'+gait2view[a]+'/'+'0000' + str(id_)+'.png'
                gender = info2[info2.ID == id_].gender.values[0]
                age = info2[info2.ID == id_].age.values[0]
                data['0000'+str(id_) + '/' + pose_view] = [pose_path, image_path, gender, age, int(id_)]

# extract gait characteristics feature
from PIL import ImageFile
import json
ImageFile.LOAD_TRUNCATED_IMAGES = True

def dotproduct(v1,v2):
    return sum((a*b) for a, b in zip(v1,v2))

def length(v):
    return math.sqrt(dotproduct(v,v))

def angle(v1,v2):
    return math.acos(dotproduct(v1,v2)/(length(v1)*length(v2)))

def rad_to_deg(x):
    return (x*180)/math.pi

def gaitFeature(pose_files):
    
    file = sorted(pose_files)
    
    shoulder_width = list()
    hip_width = list()
    spine_length = list()
    leg_length = list()
    step_width = list()
    
    step_length = list()
    knees_angle = list()
    right_hip = list()
    left_hip = list()
    right_knee = list()
    left_knee = list()
    
    for i in range(len(file)):
        with open(file[i], 'r') as f:
            json_data = json.load(f)
        h1 = np.array(eval(json.dumps(json_data['hc3d'])))
        
        shoulder_width.append(np.linalg.norm(h1[17]-h1[16], axis=0, ord=2))
        hip_width.append(np.linalg.norm(h1[2]-h1[1], axis=0, ord=2))
        spine_length.append(np.linalg.norm(h1[9]-h1[6], axis=0, ord=2) + np.linalg.norm(h1[6]-h1[3], axis=0, ord=2) + np.linalg.norm(h1[3]-h1[0], axis=0, ord=2))
        leg_length.append(np.linalg.norm(h1[1]-h1[4], axis=0, ord=2) +  np.linalg.norm(h1[4]-h1[7], axis=0, ord=2) + np.linalg.norm(h1[2]-h1[5], axis=0, ord=2) + np.linalg.norm(h1[5]-h1[8], axis=0, ord=2))
        step_width.append(abs(np.dot((h1[14]-h1[18]), (h1[1]-h1[2])/abs(h1[1]-h1[2]))))
    
        
        d = np.multiply((h1[1]-h1[2]),(h1[9]-h1[0]))
        denominator = d/abs(d)
        denominator[np.isnan(denominator)] = 0
        
        step_length.append(abs(np.dot((h1[7]-h1[8]),denominator)))
        
        
        a1 = h1[5]-h1[0]
        a2 = h1[4]-h1[0]
        knees_angle.append(rad_to_deg(angle(a1,a2)))
        
        a3 = h1[5]-h1[2]
        a4 = h1[2]-h1[0]
        right_hip.append(180-rad_to_deg(angle(a3,a4)))
        
        a5 = h1[4]-h1[1]
        a6 = h1[1]-h1[0]
        left_hip.append(180-rad_to_deg(angle(a5,a6)))
        
        a7 = h1[8]-h1[5]
        a8 = h1[5]-h1[2]
        right_knee.append(180-rad_to_deg(angle(a7,a8)))
        
        a9 = h1[7]-h1[4]
        a10 = h1[4]-h1[1]
        left_knee.append(180-rad_to_deg(angle(a9,a10)))
        
        
    return shoulder_width, hip_width, spine_length, leg_length, step_width, step_length, knees_angle, right_hip, left_hip, right_knee, left_knee

for id_view in tqdm(data.keys()):
    pose_files = data[id_view][0]
    shoulder,hip,spine,leg,step_width, step_length, knees_angle, right_hip, left_hip, right_knee, left_knee = gaitFeature(pose_files)
    
    data[id_view].append(shoulder)
    data[id_view].append(hip)
    data[id_view].append(spine)
    data[id_view].append(leg)
    data[id_view].append(step_width)
    data[id_view].append(step_length)
    data[id_view].append(knees_angle)
    data[id_view].append(right_hip)
    data[id_view].append(left_hip)
    data[id_view].append(right_knee)
    data[id_view].append(left_knee)

# extract sequnetial features
for i in tqdm(data.keys()):
    pose = []
    files = sorted(data[i][0])
    
    for file in files:
        with open(file, 'r') as f:
            json_data = json.load(f)
        pose.append(np.array(eval(json.dumps(json_data['hc3d']))).reshape(-1,))
    
    data[i].append(pose)

# encode age label
for i in data.keys():
    age = data[i][3]
    data[i][3] = int(age)
    gender = data[i][2]
    if gender == 'F':
        data[i][2] = 1
    else:
        data[i][2] = 0

# save the features as a pickle file
with open('./gait_data.pickle', 'wb') as fw:
    pickle.dump(data, fw)