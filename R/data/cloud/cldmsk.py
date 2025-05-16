#%%

import pandas as pd
import numpy as np
from pyhdf import SD
from scipy.signal import convolve2d

#%%

import sys
filename=sys.argv[1]
name=sys.argv[2]

#%%

## Load file
dat=SD.SD(filename)
coords=np.concatenate((dat.select("Longitude").get()[:,: ,None],dat.select("Latitude").get()[:,:,None]),axis=2)

## Load mask
data = dat.select("Cloud_Mask").get()[0,:,:]
cloud_mask=np.zeros(shape=data.shape+(2,))
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        val='{0:08b}'.format(data[i,j])
        cloud_mask[i,j,0]=int(val[7])
        cloud_mask[i,j,1]=int(val[6])+2*int(val[5])


## Convolution of mask values
cloud_mask[:,:,0]=convolve2d(cloud_mask[:,:,0], np.ones((3,3))/9,mode="same")
cloud_mask[:,:,1]=convolve2d(cloud_mask[:,:,1], np.ones((3,3))/9,mode="same")

## Subsample
cloud_mask=cloud_mask[np.array(range(3,cloud_mask.shape[0], 5)),:,:]
cloud_mask=cloud_mask[:,np.array(range(3,cloud_mask.shape[1], 5)),:]
cloud_mask=cloud_mask[:(coords.shape[0]),:,:]
cloud_mask=cloud_mask[:,:(coords.shape[1]),:]

## Add coordinates
cloud_mask=np.concatenate([coords,cloud_mask],axis=-1)

#%%


np.savetxt("cloud_mask_"+name+".csv",cloud_mask.reshape(-1,cloud_mask.shape[-1]),
           header="Long,Lat,IsDetermined,Mask",delimiter=",",
        	comments='')

print("Csv file saved under the name: ", "cloud_mask_"+name+".csv")

# %%
