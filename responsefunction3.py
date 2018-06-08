
# coding: utf-8

# In[ ]:

import multiprocessing
import scipy
import time
import copy
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from dipy.io.streamline import load_trk, save_trk
from dipy.tracking.streamline import Streamlines
from dipy.viz import window, actor
from dipy.viz.colormap import line_colors
from dipy.core.gradients import gradient_table
from dipy.reconst.shm import CsaOdfModel
from dipy.data import default_sphere
from dipy.data import get_sphere
from dipy.direction import peaks_from_model
from dipy.tracking import utils
from dipy.tracking.streamlinespeed import length
from dipy.reconst.shm import CsaOdfModel
from dipy.tracking.eudx import EuDX
from dipy.reconst import peaks, shm

mylabel=55 #118

#load atlas data
mypath='/Users/alex/AlexBadeaMyCodes/wenlin/Scilpy/data/'
labels_img = nib.load(mypath+'fa_labels_warp_N54917_RAS_332.nii.gz')
labels = labels_img.get_data()
print 'loading label data finished'


#load 4D image data
'loading image data'
img = nib.load(mypath+'N54917_nii4D_RAS.nii.gz')
data = img.get_data()
affine = img.affine
print 'loading img data finished'

#load bvals and bvecs data
bvals = np.loadtxt(mypath+'N54917_RAS_ecc_bvals.txt')
bvecs = np.loadtxt(mypath+'N54917_RAS_ecc_bvecs.txt')
gtab = gradient_table(bvals, bvecs)





#aoto response function in the whole brain
from dipy.reconst.csdeconv import auto_response
response1, ratio1, nvoxel = auto_response(gtab, data, roi_radius=30, fa_thr=0.7, 
                                       roi_center=None, return_number_of_voxels=True)
print  'ratio1'
print ratio1
print response1

# In[ ]:


#recursive response function
from dipy.reconst.csdeconv import recursive_response
#logic1 = np.logical_and(labels>117, labels<148)
#logic2 = np.logical_and(labels>283, labels<314)
#logic = np.logical_or(logic1, logic2)
#logic_ = np.logical_or(labels==150, labels==316)
#wm = np.where(logic, 1, np.where(logic_, 1, 0))

#wm = np.where(logic, 1,0)
response2 = recursive_response(gtab, data, mask=labels==118, sh_order=8,
                              peak_thr=0.01, init_fa=0.08,
                              init_trace=0.0021, iter=8, convergence=0.001,
                              parallel=True)

print response2

# In[ ]:


#response function from mask
from dipy.reconst.csdeconv import response_from_mask
response3, ratio3 = response_from_mask(gtab, data, mask=labels==118)

print ratio3
print response3
