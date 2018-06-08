# coding: utf-8

# In[2]:

#source activate python27
#python /Users/alex/AlexBadeaMyCodes/wenlin/connectivity_ww060718_1.py


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

'b values'
gtab.bvals

'b vecs'
gtab.bvecs

print ' gradient table calculation finished'

#Build Brain Mask
bm = np.where(labels==0, 0, 1)
mask = bm
print 'masking the brain finished'


#Compute odf in Brain Mask
print 'Start computing odf'
csamodel = shm.CsaOdfModel(gtab, 6) # spherical hamronics order 6
start_time = time.time()

#sphere = get_sphere('symmetric724')
csapeaks = peaks.peaks_from_model(model=csamodel,
                                  data=data,
                                  sphere=peaks.default_sphere,
                                  #sphere=sphere,
                                  relative_peak_threshold=.1,
                                  min_separation_angle=25,
                                  mask=mask,
                                  parallel=True,
                                  sh_order=8,
                                  npeaks=5)
time_parallel = time.time() - start_time
print("peaks_from_model using " + str(multiprocessing.cpu_count())
      + " process ran in :" + str(time_parallel) + " seconds")



#ren = window.Renderer()
#ren.add(actor.peak_slicer(csd_peaks.peak_dirs,
#                          csd_peaks.peak_values,
#                          colors=None))

#if interactive:
#    window.show(ren, size=(900, 900))
#else:
#    window.record(ren, out_path='csd_direction_field.png', size=(900, 900))
    


print 'Start computing streamlines'
start_time2 = time.time()
#Compute Streamlines
myseeds = utils.seeds_from_mask(labels==mylabel, density=2)
#myseeds=1000000
#streamline_generator = EuDX(csapeaks.peak_values, csapeaks.peak_indices,
#                            odf_vertices=peaks.default_sphere.vertices,
#                            a_low=.05, step_sz=.5, seeds=myseeds)

streamline_generator = EuDX(csapeaks.peak_values, csapeaks.peak_indices,
                            odf_vertices=peaks.default_sphere.vertices,
                            a_low=0.05, step_sz=.5, seeds=myseeds, ang_thr=70.0,
                            length_thr=0.0, total_weight=0.5, max_points=1000, affine=None)

#step_size is in voxels

time_parallel = time.time() - start_time2
print("streamlines_EuDX " +str(myseeds)+  str(multiprocessing.cpu_count())
      + " process ran in :" + str(time_parallel) + " seconds")
affine = streamline_generator.affine

streamlines = Streamlines(streamline_generator)

# save streamlines
save_trk(mypath+str(mylabel)+"EuDX.trk", streamlines, affine,
         shape=labels.shape,
         vox_size=labels_img.header.get_zooms())

#Build Target
# cc_slice = labels == 2
# cc_streamlines = utils.target(streamlines, cc_slice, affine=affine)
# cc_streamlines = Streamlines(cc_streamlines)

# other_streamlines = utils.target(streamlines, cc_slice, affine=affine,
#                                  include=False)
# other_streamlines = Streamlines(other_streamlines)
# assert len(other_streamlines) + len(cc_streamlines) == len(streamlines)

# Enables/disables interactive visualization
interactive = True

print 'displaying results'
# Make display objects
color = line_colors(streamlines)
streamlines_actor = actor.line(streamlines, line_colors(streamlines))
# cc_ROI_actor = actor.contour_from_roi(cc_slice, color=(1., 1., 0.),
#                                       opacity=0.5)

# vol_actor = actor.slicer(t1_data)

# vol_actor.display(x=40)
# vol_actor2 = vol_actor.copy()
# vol_actor2.display(z=35)

# Add display objects to canvas
r = window.Renderer()
# r.add(vol_actor)
# r.add(vol_actor2)
r.add(streamlines_actor)
# r.add(cc_ROI_actor)



#Build the connectivity matrix
M, grouping = utils.connectivity_matrix(streamlines, labels, affine=affine,
                                        return_mapping=True,
                                        mapping_as_streamlines=True)
np.savetxt(mypath+str(mylabel)+"connectivity matrix.csv", M, delimiter = ',')
plt.imshow(np.log1p(M), interpolation='nearest')
plt.savefig(mypath+str(mylabel)+"connectivity.png")


# Save figures
window.record(r, n_frames=1, out_path=mypath+str(mylabel)+'_streamlines.png', size=(800, 800))
if interactive:
    window.show(r)
# r.set_camera(position=[-1, 0, 0], focal_point=[0, 0, 0], view_up=[0, 0, 1])
# window.record(r, n_frames=1, out_path='corpuscallosum_sagittal.png',
#               size=(800, 800))
# if interactive:
#     window.show(r)


