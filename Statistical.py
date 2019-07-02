import os
import numpy as np
import pywt
import sklearn.preprocessing as pre
import cv2
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

Benign = []
folder = 'D:/MIAS/Benign'
for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename),0)# Reading images from a folder
        if img is not None:
            Benign.append(img)
        

# MALIGNENT PART

Malignant = []
folder = 'D:/MIAS/Malignant'
for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename),0) # Reading images from a folder
        if img is not None:
            Malignant.append(img)
        


# Normal cases

Normal = []
folder = 'D:/MIAS/Normal'
for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename),0) # Reading images from a folder
        if img is not None:
            Normal.append(img)





#Applying DWT and extracting features on a Benign dataset
#Block Size of  8 by 8
# Define the window size
windowsize_r =8
windowsize_c =8

Benign_data=[]
for i in range(len(Benign)):
    coeffs=pywt.dwt2(Benign[i],'db1')

    features=[]


    #calculating statistical features  for Approximation subband
    win=[]
    range_x=coeffs[0].shape[1]
    range_y=coeffs[0].shape[0]
    for r in range(0,range_x,8):
        for c in range(0,range_y,8):
            window = coeffs[0][r:r+windowsize_r,c:c+windowsize_c]
            win.append(window)
    for i in range(len(win)):
        mean=np.mean(win[i])
        features.append(mean)
        std_devi=np.std(win[i])
        features.append(std_devi)
        energy=(win[i]**2).sum()/64
        features.append(energy)
    

    #caluclating statistical feature for horizontal detail(cH) subband
    win=[]
    for r in range(0,range_x,8):
        for c in range(0,range_y,8):
            window = coeffs[1][0][r:r+windowsize_r,c:c+windowsize_c]
            win.append(window)
    for i in range(len(win)):
        mean=np.mean(win[i])
        features.append(mean)
        std_devi=np.std(win[i])
        features.append(std_devi)
        energy=(win[i]**2).sum()/64
        features.append(energy)
 
                 
    #CALULATING statistical features for vertical detail(cV)  subband
    win=[]
    for r in range(0,range_x,8):
        for c in range(0,range_y,8):
            window = coeffs[1][1][r:r+windowsize_r,c:c+windowsize_c]
            win.append(window)
    for i in range(len(win)):
        mean=np.mean(win[i])
        features.append(mean)
        std_devi=np.std(win[i])
        features.append(std_devi)
        energy=(win[i]**2).sum()/64
        features.append(energy)
     #calculating statistical features for  diagonal detail(cD) subband
    win=[]
    for r in range(0,range_x,8):
        for c in range(0,range_y,8):
            window = coeffs[1][2][r:r+windowsize_r,c:c+windowsize_c]
            win.append(window)
    for i in range(len(win)):
        mean=np.mean(win[i])
        features.append(mean)
        std_devi=np.std(win[i])
        features.append(std_devi)
        energy=(win[i]**2).sum()/64
        features.append(energy)
    Benign_data.append(features)
Benign_data=np.asarray(Benign_data)
out=np.asarray([np.ones(len(Benign))])   #ADDING the label in the last column
Benign_data=np.concatenate((Benign_data,out.T),axis=1)
print(len(Benign))    
print(Benign_data.shape)

#Applying DWT and extracting features on a Malignant dataset


Malignant_data=[]
for i in range(len(Malignant)):
    coeffs=pywt.dwt2(Malignant[i],'db1')

    features=[]

    #calculating statistical features  for Approximation subband
    win=[]
    range_x=coeffs[0].shape[1]
    range_y=coeffs[0].shape[0]
    for r in range(0,range_x,8):
        for c in range(0,range_y,8):
            window = coeffs[0][r:r+windowsize_r,c:c+windowsize_c]
            win.append(window)
    for i in range(len(win)):
        mean=np.mean(win[i])
        features.append(mean)
        std_devi=np.std(win[i])
        features.append(std_devi)
        energy=(win[i]**2).sum()/64
        features.append(energy)
     

    #caluclating statistical feature for horizontal detail(cH) subband
    win=[]
    for r in range(0,range_x,8):
        for c in range(0,range_y,8):
            window = coeffs[1][0][r:r+windowsize_r,c:c+windowsize_c]
            win.append(window)
    for i in range(len(win)):
        mean=np.mean(win[i])
        features.append(mean)
        std_devi=np.std(win[i])
        features.append(std_devi)
        energy=(win[i]**2).sum()/64
        features.append(energy)
 
                 
    #CALULATING statistical features for vertical detail(cV)  subband
    win=[]
    for r in range(0,range_x,8):
        for c in range(0,range_y,8):
            window = coeffs[1][1][r:r+windowsize_r,c:c+windowsize_c]
            win.append(window)
    for i in range(len(win)):
        mean=np.mean(win[i])
        features.append(mean)
        std_devi=np.std(win[i])
        features.append(std_devi)
        energy=(win[i]**2).sum()/64
        features.append(energy)
    #calculating statistical features for  diagonal detail(cD) subband
    win=[]
    for r in range(0,range_x,8):
        for c in range(0,range_y,8):
            window = coeffs[1][2][r:r+windowsize_r,c:c+windowsize_c]
            win.append(window)
    for i in range(len(win)):
        mean=np.mean(win[i])
        features.append(mean)
        std_devi=np.std(win[i])
        features.append(std_devi)
        energy=(win[i]**2).sum()/64
        features.append(energy)
    Malignant_data.append(features)
Malignant_data=np.asarray(Malignant_data)
out=np.asarray([2*np.ones(len(Malignant))])     #ADDING the label                                            
Malignant_data=np.concatenate((Malignant_data,out.T),axis=1)
print(len(Malignant))    
print(Malignant_data.shape)


#Applying DWT and extracting features on NORMAL dataset


Normal_data=[]
for i in range(len(Normal)):
    coeffs=pywt.dwt2(Normal[i],'db1')

    features=[]


    #calculating statistical features  for Approximation subband
    win=[]
    range_x=coeffs[0].shape[1]
    range_y=coeffs[0].shape[0]
    for r in range(0,range_x,8):
        for c in range(0,range_y,8):
            window = coeffs[0][r:r+windowsize_r,c:c+windowsize_c]
            win.append(window)
    for i in range(len(win)):
        mean=np.mean(win[i])
        features.append(mean)
        std_devi=np.std(win[i])
        features.append(std_devi)
        energy=(win[i]**2).sum()/64
        features.append(energy)
     

    #caluclating statistical feature for horizontal detail(cH) subband
    win=[]
    for r in range(0,range_x,8):
        for c in range(0,range_y,8):
            window = coeffs[1][0][r:r+windowsize_r,c:c+windowsize_c]
            win.append(window)
    for i in range(len(win)):
        mean=np.mean(win[i])
        features.append(mean)
        std_devi=np.std(win[i])
        features.append(std_devi)
        energy=(win[i]**2).sum()/64
        features.append(energy)
 
                 
    #CALULATING statistical features for vertical detail(cV)  subband
    win=[]
    for r in range(0,range_x,8):
        for c in range(0,range_y,8):
            window = coeffs[1][1][r:r+windowsize_r,c:c+windowsize_c]
            win.append(window)
    for i in range(len(win)):
        mean=np.mean(win[i])
        features.append(mean)
        std_devi=np.std(win[i])
        features.append(std_devi)
        energy=(win[i]**2).sum()/64
        features.append(energy)
    #calculating statistical features for  diagonal detail(cD) subband
    win=[]
    for r in range(0,range_x,8):
        for c in range(0,range_y,8):
            window = coeffs[1][2][r:r+windowsize_r,c:c+windowsize_c]
            win.append(window)
    for i in range(len(win)):
        mean=np.mean(win[i])
        features.append(mean)
        std_devi=np.std(win[i])
        features.append(std_devi)
        energy=(win[i]**2).sum()/64
        features.append(energy)
    Normal_data.append(features)
Normal_data=np.asarray(Normal_data)
out=np.asarray([np.zeros(len(Normal))])   #ADDING the output result for the data in the last column
Normal_data=np.concatenate((Normal_data,out.T),axis=1)
print(len(Normal))    
print(Normal_data.shape)

train_data=np.concatenate((Malignant_data,Benign_data,Normal_data),axis=0)
train_X=train_data[:,:-1]
train_y=train_data[:,-1]

#third=(len(Malignant)//3)
3#3Malignant_test=Malignant_data[-third:,:]
#Malignant_train=Malignant_data[:-third,:]

#third=(len(Benign)//3)
#Benign_test=Benign_data[-third:,:]
#Benign_train=Benign_data[:-third,:]

#third=(len(Normal)//3)
#Normal_test=Normal_data[-third:,:]
#Normal_train=Normal_data[:-third,:]

#train_data=np.concatenate((Malignant_train,Benign_train,Normal_train),axis=0)
#test_data=np.concatenate((Malignant_test,Benign_test,Normal_test),axis=0)



#np.random.shuffle(train_data)
#np.random.shuffle(test_data)



#X_train=train_data[:,:-1]
# Y_train=train_data[:,-1]
# X_test=test_data[:,:-1]
# Y_test=test_data[:,-1]

# X= SelectKBest(chi2, k=5).fit(train_X,train_y)
# X_new=X.transform(train_X)

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(train_X,train_y, test_size=0.3, random_state=42)




#Normalizing Training and Testing dataset
X_train=pre.scale(X_train)
X_test=pre.scale(X_test)

print("Training data shape is:",X_train.shape)
print("Test data shape is:",X_test.shape)
print(y_test)
