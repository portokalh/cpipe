
# coding: utf-8

# In[2]:


import time
import copy
import numpy as np
import nibabel as nib
import dipy.reconst.dti as dti
from dipy.viz import window, actor
from dipy.viz.colormap import line_colors
from dipy.core.gradients import gradient_table


# In[3]:


st_all_animals = time.time()
roi_l = 122
roi_r = 1122+166-1000
interactive = True
l=['N54717','N54718','N54719','N54720','N54722','N54759','N54760','N54761','N54762','N54763','N54764','N54765','N54766','N54770','N54771','N54772','N54798','N54801','N54802','N54803','N54804','N54805','N54806','N54807','N54818','N54824','N54825','N54826','N54837','N54838','N54843','N54844','N54856','N54857','N54858','N54859','N54860','N54861','N54873','N54874','N54875','N54876','N54877','N54879','N54880','N54891','N54892','N54893','N54897','N54898','N54899','N54900','N54915','N54916','N54917']
for j in range(54,55):
    print(j)
    runno = l[j]

    mypath = '/Users/wenlin_wu/Summer Research/Scilpy/data/' # 'data/'
    outpath = mypath + 'results_0626/' #  'results_0610/'

    #load 4D image data
    st_all = time.time()
    img = nib.load('/Users/wenlin_wu/Summer Research/Scilpy/data/N54917_nii4D_RAS.nii')
    data = img.get_data()
    affine = img.affine
    print 'loading img data finished'

    #load atlas data
    labels_img = nib.load(mypath+'fa_labels_warp_'+runno+'_RAS.nii.gz')
    labels = labels_img.get_data()
    print 'loading label data finished'

    #transform label number to make it 1~332
    labels_ = copy.copy(labels)
    nonz = np.nonzero(labels)
    for i in range(len(nonz[0])):
        if labels_[nonz[0][i], nonz[1][i], nonz[2][i]] >= 1000:
            labels_[nonz[0][i], nonz[1][i], nonz[2][i]] -= 1000
            labels_[nonz[0][i], nonz[1][i], nonz[2][i]] += 166
    print 'label transformation finished'

    #load bvals and bvecs data
    bvals = np.loadtxt(mypath+runno+'_RAS_ecc_bvals.txt')
    bvecs = np.loadtxt(mypath+runno+'_RAS_ecc_bvecs.txt')
    #bvecs[:, [1,2]] = bvecs[:,[2,1]]
    gtab = gradient_table(bvals, bvecs)
    print 'gradient table calculation finished'
    print(gtab.info)

    #Build Brain Mask
    #bm = np.where(labels==0, False, True)
    #print 'masking the brain finished'
    
    # white matter mask
    logic1 = np.logical_and(labels_>117, labels_<148)
    logic2 = np.logical_and(labels_>283, labels_<314)
    logic = np.logical_or(logic1, logic2)
    logic_ = np.logical_or(labels_==150, labels_==316)
    wm = np.where(logic, True, np.where(logic_, True, False))

    mask = wm


# In[4]:


tenmodel = dti.TensorModel(gtab)


# In[5]:


bm = np.where(labels==0, False, True)
tenfit = tenmodel.fit(data, bm)


# In[6]:


from dipy.reconst.dti import fractional_anisotropy, color_fa

FA = fractional_anisotropy(tenfit.evals)


# In[7]:


FA[np.isnan(FA)] = 0
fa_img = nib.Nifti1Image(FA.astype(np.float32), img.affine)
nib.save(fa_img, 'bmtensor_fa.nii.gz')


# In[8]:


FA = np.clip(FA, 0, 1)
RGB = color_fa(FA, tenfit.evecs)
nib.save(nib.Nifti1Image(np.array(255 * RGB, 'uint8'), img.affine), 'bmtensor_rgb.nii.gz')


# In[1]:


from dipy.data import get_sphere
sphere = get_sphere('symmetric724')

from dipy.viz import window, actor

# Enables/disables interactive visualization
interactive = False

ren = window.Renderer()

evals = tenfit.evals
evecs = tenfit.evecs
cfa = RGB
cfa /= cfa.max()

ren.add(actor.tensor_slicer(evals, evecs, scalar_colors=cfa, sphere=sphere, scale=0.3))

print('Saving illustration as tensor_ellipsoids.png')
window.record(ren, n_frames=1, out_path='tensor_ellipsoids.png', size=(600, 600))

