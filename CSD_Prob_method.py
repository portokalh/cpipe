
# coding: utf-8

# In[15]:


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
from dipy.data import default_sphere
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)
from dipy.tracking import utils
from dipy.tracking.streamlinespeed import length
from dipy.reconst import peaks, shm
from dipy.tracking.local import LocalTracking, BinaryTissueClassifier
from dipy.direction import ProbabilisticDirectionGetter
from dipy.reconst.csdeconv import recursive_response


# In[17]:

runno='N54917'

#load all the data
#load atlas data

#mypath='data/'
mypath='/Users/alex/brain_data/E3E4/wenlin/4Dnii/' # 4Dnii, bvec, bavules live here too
mypathlabels='/Users/alex/brain_data/E3E4/wenlin/labels/'
mypathout='/Users/alex/brain_data/E3E4/wenlin/results/'

#labels_img = nib.load(mypathlabels+'fa_labels_warp_'+runno+'_RAS_332.nii.gz')

labels_img = nib.load(mypathlabels+'fa_labels_warp_'+runno+'_RAS.nii.gz')
labels = labels_img.get_data()
print 'loading label data finished'

labels_ = copy.copy(labels)
nonz = np.nonzero(labels)
for i in range(len(nonz[0])):
    if labels_[nonz[0][i], nonz[1][i], nonz[2][i]] >= 1000:s
#load 4D image data
img = nib.load(mypath+runno+'_nii4D_RAS.nii.gz')
data = img.get_data()
affine = img.affine
print 'loading img data finished'

#load bvals and bvecs data
bvals = np.loadtxt(mypath+runno+'_RAS_ecc_bvals.txt')
bvecs = np.loadtxt(mypath+runno+'_RAS_ecc_bvecs.txt')
gtab = gradient_table(bvals, bvecs)
print 'gradient table calculation finished'

#Build Brain Mask
bm = np.where(labels==0, False, True)
#bm = np.where(labels==121,True,False)
mask = bm

seeds = utils.seeds_from_mask(bm, density=[2, 2, 2], affine=affine)

print 'masking the brain finished'

# white matter mask
# logic1 = np.logical_and(labels>117, labels<148)
# logic2 = np.logical_and(labels>283, labels<314)
# logic = np.logical_or(logic1, logic2)
# logic_ = np.logical_or(labels==150, labels==316)
# wm = np.where(logic, True, np.where(logic_, True, False))


# In[18]:


#calculating response function and fitting csd model
start_time1 = time.time()

#response, ratio = auto_response(gtab, data, roi_radius=20, fa_thr=0.7)

response = recursive_response(gtab, data, mask=labels==121, sh_order=8,
                              peak_thr=0.01, init_fa=0.08,
                              init_trace=0.0021, iter=8, convergence=0.001,
                              parallel=True)


csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
csd_fit = csd_model.fit(data, mask=bm)

time_odf = time.time() - start_time1
print("computing odf using " + str(multiprocessing.cpu_count())
      + " process ran in :" + str(time_odf) + " seconds")


# In[23]:


#get direction from csd model
print 'get directions from model'
start_time2 = time.time()
prob_dg = ProbabilisticDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
                                                    max_angle=30.,
                                                    sphere=default_sphere)
print 'binary classfier'
#use binary classifier
classifier = BinaryTissueClassifier(bm)

#local tracking
print 'local tracking'
streamlines_generator = LocalTracking(prob_dg, classifier, seeds, 
                                    affine, step_size=.5, max_cross=1)
affine = streamlines_generator.affine

print 'streamlines now'
streamlines = Streamlines(streamlines_generator)

time_stream = time.time() - start_time2
print("computing streamlines using " + str(multiprocessing.cpu_count())
      + " process ran in :" + str(time_stream) + " seconds")

# save streamlines
save_trk(mypathout+"CSDProb.trk", streamlines, affine,
         shape=labels.shape,
         vox_size=labels_img.header.get_zooms())

# Computing Connectivity Matrix and save csv
M, grouping = utils.connectivity_matrix(streamlines, labels_new, affine=affine,
                                        return_mapping=False,
                                        mapping_as_streamlines=)
np.savetxt(mypathout+runno+"CSDProb_connectivity.csv", M, delimiter = ',')


# In[24]:


#Visualize the connectivity matrix
plt.imshow(np.log1p(M), interpolation='nearest')
plt.savefig(mypath+runno+"CSDProb_connectivity.png")

#Visualize the streamlines
interactive = False
color = line_colors(streamlines)
streamlines_actor = actor.line(streamlines, line_colors(streamlines))
r = window.Renderer()
r.add(streamlines_actor)

window.record(r, n_frames=1, out_path=mypathout+runno+'CSDProbstreamlines.png',
              size=(800, 800))

if interactive:
    window.show(r)

