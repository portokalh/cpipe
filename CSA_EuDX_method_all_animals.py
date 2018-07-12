
# coding: utf-8

# In[1]:


from __future__ import division
import multiprocessing
import scipy
import time
import copy
import gc
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from collections import Counter

from dipy.data import get_sphere, small_sphere, default_sphere
from dipy.io.streamline import load_trk, save_trk
from dipy.tracking.streamline import Streamlines
from dipy.viz import window, actor
from dipy.viz.colormap import line_colors
from dipy.core.gradients import gradient_table
from dipy.reconst.shm import CsaOdfModel
from dipy.tracking.eudx import EuDX

from dipy.tracking import utils
from dipy.tracking.streamlinespeed import length
from dipy.reconst import peaks, shm
from dipy.tracking.local import BinaryTissueClassifier
from dipy.direction import ProbabilisticDirectionGetter
from dipy.direction import peaks_from_model
from dipy.tracking.streamlinespeed import length



print('11 July 2018 uses 20000 seeds from whole brain')
print('Main Information\nthis script we use CSA model along with EuDX tracking method to generate streamlines.\nHere is the parameter information:')
print('\033[1m%s \033[0m'%'affine:')
print('(4,4) identity matrix')
print('\033[1m%s \033[0m'%'peaks_from_model:')
print('sphere = get_sphere(symmetric724), relative_peak_threshold=0.1, min_separation_angle=25, mask = bm\n')
print('\033[1m%s \033[0m'%'ProbabilisticDirectionGetter.from_shcoeff:')
print('max_angle=70.,relative_peak_threshold=0.1,sphere=default_sphere\n')
print('\033[1m%s \033[0m'%'EuDX:')
print('random seeds from brain mask and brain mask classifier\n')
print('Note: We also filter out streamlines length≤3')


st_all_animals = time.time()
l=['N54900','N54717','N54718','N54719','N54720','N54722','N54759','N54760','N54761','N54762','N54763','N54764','N54765','N54766','N54770','N54771','N54772','N54798','N54801','N54802','N54803','N54804','N54805','N54806','N54807','N54818','N54824','N54825','N54826','N54837','N54838','N54843','N54844','N54856','N54857','N54858','N54859','N54860','N54861','N54873','N54874','N54875','N54876','N54877','N54879','N54880','N54891','N54892','N54893','N54897','N54898','N54899','N54900','N54915','N54916','N54917']

for j in range(55):
    print(j)
    runno = l[j]

    mypath = '/Users/alex/brain_data/E3E4/wenlin/' # 'data/'
    outpath = mypath + 'results/' #  'results_0610/'

    #load 4D image data
    st_all = time.time()
    img = nib.load(mypath+'4Dnii/'+runno+'_nii4D_RAS.nii.gz')
    data = img.get_data()
    affine = np.diag(np.ones(4))
    print 'loading img data finished'

    #load atlas data
    labels_img = nib.load(mypath+'labels/'+'fa_labels_warp_'+runno+'_RAS.nii.gz')
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
    bvals = np.loadtxt(mypath+'4Dnii/'+runno+'_RAS_ecc_bvals.txt')
    bvecs = np.loadtxt(mypath+'4Dnii/'+runno+'_RAS_ecc_bvecs.txt')
    bvecs = np.c_[bvecs[:,0],bvecs[:,1],-bvecs[:,2]]
    gtab = gradient_table(bvals, bvecs)
    print 'gradient table calculation finished'
    print(gtab.info)

    #Build Brain Mask
    bm = np.where(labels==0, False, True)
    mask = bm
    print 'masking the brain finished'

    # # white matter mask
    # logic1 = np.logical_and(labels_>117, labels_<148)
    # logic2 = np.logical_and(labels_>283, labels_<314)
    # logic = np.logical_or(logic1, logic2)
    # logic_ = np.logical_or(labels_==150, labels_==316)
    # wm = np.where(logic, True, np.where(logic_, True, False))



    print 'fitting CSA model'
    st2 = time.time()
    csamodel = shm.CsaOdfModel(gtab, 6)
    csapeaks = peaks.peaks_from_model(model=csamodel,
                                       data=data,
                                       sphere=peaks.default_sphere,
                                       relative_peak_threshold=.1,
                                       min_separation_angle=25,
                                       mask=wm,
                                       sh_order=8,
                                       npeaks=5,
                                       parallel=True)
    et2 = time.time() - st2
    print 'fitting CSA model finished, running time is {}'.format(et2)


    # plot peaks
    # interactive = False
    # ren = window.Renderer()
    # slice_actor = actor.peak_slicer(csd_peaks.peak_dirs,
    #                           csd_peaks.peak_values,
    #                           colors=None)
    # slice_actor.RotateX(90)

    # ren.add(slice_actor)
    # if interactive:
    #     window.show(ren, size=(900, 900))
    # else:
    #     ren.set_camera(position=[0,-1,0], focal_point=[0,0,0], view_up=[0,0,1])
    #     window.record(ren, out_path='csd_direction_bm.png', size=(900, 900))



    print 'seeding begins, using np.random.seed(123)'
    st3 = time.time()
    np.random.seed(123)
    #seeds = utils.random_seeds_from_mask(mask, 1) #does crash because of memory limitations
    seeds = utils.random_seeds_from_mask(mask, 20000, seed_count_per_voxel=False)
    for i in range(len(seeds)):
            if seeds[i][0]>199.:
                seeds[i][0]=398-seeds[i][0]
            if seeds[i][1]>399.:
                seeds[i][1]=798-seeds[i][1]
            if seeds[i][2]>199.:
                seeds[i][2]=398-seeds[i][2]
            for j in range(3):
                if seeds[i][j]<0.:
                    seeds[i][j]=-seeds[i][j]
    et3 = time.time() - st3
    print 'seeding transformation finished, the total seeds are {}, running time is {}'.format(seeds.shape[0], et3)

    print 'generating streamlines begins'
    st4 = time.time()
    del data, img, labels, labels_img
    gc.collect()
    print 'data, img, labels, labels_img elete to save memory'

    streamline_generator = EuDX(csapeaks1.peak_values, csapeaks1.peak_indices,
                                odf_vertices=peaks.default_sphere.vertices,
                                a_low=.05, step_sz=.5, seeds=seeds, ang_thr=70.0,length_thr=0.0,
                                total_weight=0.5,max_points=1000,affine=None)
    affine = streamline_generator.affine

    streamlines = Streamlines(streamline_generator)
    et4 = time.time() - st4
    lengths = [length(sl).astype(np.int) for sl in streamlines]
    print 'generating streamlines finished, the length is {}~{}, running time is {}'.format(np.min(lengths), np.max(lengths), et4)

    del bm, mask, lengths
    gc.collect()
    print 'bm, mask, lengths delete to save memory'

    #Cut streamlines
    streamlines = [sl for sl in streamlines if length(sl).astype(np.int)>3]
    print 'we get {} streamlines'.format(len(streamlines))
    print 'cutting short streamlines finished'

    save_trk(outpath+'connectivity_csv/'+runno+'_CSAstreamlines.trk', streamlines=streamlines, affine=np.eye(4))
    print 'streamlines saved'

    print 'building connectivity matrix begins'
    st5 = time.time()
    M= utils.connectivity_matrix(streamlines, labels_, affine=affine,
                                            return_mapping=False,
                                            mapping_as_streamlines=False)
    del streamlines
    gc.collect()
    print 'streamlines delete to save memory'

    M = M[1:, 1:]
    et5 = time.time() - st5
    np.savetxt(outpath+'connectivity_csv/'+runno+'_connectivitybm.csv', M, delimiter = ',')
    print(runno+'connectivity matrix csv saved, the running time is {}'.format(et5))

    del M
    gc.collect()

    et_all = time.time() - st_all
    print(runno+' reconstruction finished, running time is {}'.format(et_all))

et_all_animals = time.time() - st_all_animals
print 'all animals reconstruction finished, we reconstruct connectivity matrix for {} animals, running time is {}'.format(len(l), et_all_animals)
