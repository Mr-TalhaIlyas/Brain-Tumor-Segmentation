import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "1";
import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from model import U_net, BU_net
from custom_loss import Weighted_BCEnDice_loss
from custom_metric import mean_iou, dice_coef
# Loading the data
X_train = np.load('/home/user01/data_ssd/Talha/brats/X_train_norm_resized.npy')
X_valid = np.load('/home/user01/data_ssd/Talha/brats/X_test_norm_resized.npy') 
Y_train = np.load('/home/user01/data_ssd/Talha/brats/Y_train_norm_resized.npy') 
Y_valid = np.load('/home/user01/data_ssd/Talha/brats/Y_test_norm_resized.npy') 
 


#%%
im_width = 256
im_height = 256
batch_size = 8
input_img = Input((im_height, im_width, 4), name='img')
model = BU_net(input_img, n_filters=64, dropout=0.3, batchnorm=True)
model.compile(optimizer=Adam(), loss = Weighted_BCEnDice_loss, metrics = [mean_iou, dice_coef])
#model.summary()
callbacks = [
    EarlyStopping(patience=10, verbose=1), #Early Stopping if loss is not reduced since 10 Epochs
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),  #Decay LR if loss is not reduced since 5 Epochs
    #ModelCheckpoint('D:/Anaconda/Keras/Segmentation/test_cityscape.h5', verbose=1, save_best_only=True, save_weights_only=True) # Save weights if val loss is improved
]
#%%
results = model.fit(X_train, Y_train, batch_size=batch_size, epochs=50, callbacks=callbacks,validation_data=(X_valid, Y_valid))
#%%    Evaluate and predict
# load the best model
model.load_weights('/home/user01/data_ssd/simpleUweighted_bce.h5')
#%%
model.evaluate(X_valid, Y_valid, verbose=1)# gives accuracy and loss as specified

#%%        Plotting                                   Plotting

f, ax = plt.subplots(2, 2, figsize = (10,10), sharex=False)
        
x = np.arange(len(results.history["val_loss"])) + 1 

#ax1.set_yscale('log')
ax[0,0].plot(x, results.history["loss"], 'g--*', label="loss")
ax[0,0].plot(x, results.history["val_loss"], 'r-.+', label="val_loss")
ax[0,0].legend()

ax[0,1].plot(x, results.history["mean_iou"], 'g--*', label='iou')
ax[0,1].plot(x, results.history["val_mean_iou"], 'r-.+', label="val_iou")
ax[0,1].legend()

ax[1,0].plot(x, results.history["dice_coef"], 'g--*', label='dice_coef')
ax[1,0].plot(x, results.history["val_dice_coef"], 'r-.+', label="val_dice_coef")
ax[1,0].legend()


plt.show();
#%% Plotting Results 
index = 18
temp1 = X_train[index]
temp2 = temp1[np.newaxis,:,:,:]

preds_train = np.squeeze(model.predict(temp2, verbose=1))
preds_train = (preds_train > 0.2).astype(np.uint8)

#%
fig, ax = plt.subplots(2, 4, figsize = (20, 10))

ax[0,0].imshow(temp1[...,0], cmap = 'gray', interpolation = 'bilinear')
ax[0,0].axis("off")
ax[0,0].set_title('Scan_1')

ax[0,1].imshow(temp1[...,1], cmap = 'gray', interpolation = 'bilinear')
ax[0,1].axis("off")
ax[0,1].set_title('Scan_2')

ax[0,2].imshow(temp1[...,2], cmap = 'gray', interpolation = 'bilinear')
ax[0,2].axis("off")
ax[0,2].set_title('Scan_3')

ax[0,3].imshow(temp1[...,3], cmap = 'gray', interpolation = 'bilinear')
ax[0,3].axis("off")
ax[0,3].set_title('Scan_4')

ax[1,0].imshow(preds_train[..., 1], cmap = 'gray', interpolation = 'bilinear')
ax[1,0].axis("off")
ax[1,0].set_title('TC1')

ax[1,1].imshow(preds_train[..., 2], cmap = 'gray', interpolation = 'bilinear')
ax[1,1].axis("off")
ax[1,1].set_title('TC2')

ax[1,2].imshow(preds_train[..., 3], cmap = 'gray', interpolation = 'bilinear')
ax[1,2].axis("off")
ax[1,2].set_title('TC1_e')

ax[1,3].imshow(preds_train[..., 4], cmap = 'gray', interpolation = 'bilinear')
ax[1,3].axis("off")
ax[1,3].set_title('Flair')
#%%
alpha = 0.5
beta = (1.0 - alpha)
# Combined overlayed results
ip = X_train[index,:,:,0]
ip = cv2.merge((ip, ip, ip)) # make 3 channels for merging
gt = Y_train[index,:,:,2:5].astype(np.float64)
preds = preds_train[:,:,2:5].astype(np.float64)

o_gt = cv2.addWeighted(ip, alpha, gt, beta, 0.0)
o_pred = cv2.addWeighted(ip, alpha, preds, beta, 0.0)

fig, ax = plt.subplots(1, 3, figsize = (10, 5))

ax[0].imshow(ip)
ax[0].axis("off")
ax[0].set_title('Input')

ax[1].imshow(o_gt)
ax[1].axis("off")
ax[1].set_title('GT')

ax[2].imshow(o_pred)
ax[2].axis("off")
ax[2].set_title('Pred')


#%%
import cv2
tempz= preds_train.squeeze()*255
cv2.imwrite('D:/Anaconda/Image_Processor/seg.jpg', cv2.cvtColor(tempz,cv2.COLOR_BGR2RGB))