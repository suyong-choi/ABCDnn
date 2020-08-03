from ABCD_dnn_mmd import ABCDdnn
import numpy as np
import matplotlib.pyplot as plt
import uproot
from onehotencoder import OneHotEncoder_int
from skhep.visual import MplPlotter
import os
import pandas as pd

featurevars = ['met', 'ht', 'pt5', 'pt6', 'njet', 'nbtag']

rootfile='ttjjresult.root'

def prepdata():
    ttjj = uproot.open(rootfile)
    ttjjtree = ttjj['mytree']
    iscategorical = [False, False, False, False, True, True]
    upperlimit = [10, 10, 10, 10, 9, 3]
    _onehotencoder = OneHotEncoder_int(iscategorical, upperlimit=upperlimit)

    inputtmp = ttjjtree.pandas.df(featurevars)

    iscategorical = np.array(inputtmp.dtypes == np.int32)

    inputnumpy = inputtmp.to_numpy(dtype=np.float32)
    inputs = _onehotencoder.encode(inputnumpy)
    ncats = _onehotencoder.ncats
    ncat_per_feature = _onehotencoder.categories_per_feature

    meanslist = []
    sigmalist = []
    currentcolumn = 0
    for ifeat, ncatfeat in zip(range(inputtmp.shape[1]), ncat_per_feature):
        if ncatfeat == 0: # fir float features, get mean and sigma
            mean = np.mean(inputnumpy[:, currentcolumn], axis=0, dtype=np.float32).reshape(1,1)
            meanslist.append(mean)
            sigma = np.std(inputnumpy[:, currentcolumn], axis=0, dtype=np.float32).reshape(1,1)
            sigmalist.append(sigma)
            currentcolumn += 1
        else: # categorical features do not get changed
            mean = np.zeros(shape=(1, ncatfeat), dtype=np.float32) 
            meanslist.append(mean)
            sigma = np.ones(shape=(1, ncatfeat), dtype=np.float32)
            sigmalist.append(sigma)
            currentcolumn += ncatfeat

    inputmeans = np.hstack(meanslist)
    inputsigma = np.hstack(sigmalist)

    normedinputs = (inputs-inputmeans) / inputsigma

    return inputtmp, normedinputs, inputmeans, inputsigma, ncat_per_feature

def writetorootfile(rootfilename, datadict):
    branchdict = {}
    for key, data in datadict.items():
        branchdict[key] = data.dtype
    tree = uproot.newtree(branches=branchdict)
    with uproot.recreate(rootfilename) as f:
        f['mytree'] = tree
        f['mytree'].extend(datadict)

    pass

def train_and_validate(steps=10000, minibatch=128, LRrange=[0.0001, 0.00001, 10000, 0], beta1=0.9, beta2=0.999, nafdim=16, depth=2, \
    savedir='abcdnn', seed=100, retrain=False, train=True):
    rawinputs, normedinputs, inputmeans, inputsigma, ncat_per_feature = prepdata()
    print(ncat_per_feature)
    inputdim = 4
    ncat_per_feature = ncat_per_feature[0:inputdim]
    conddim = normedinputs.shape[1] - inputdim

    issignal = (rawinputs['njet']>=9) & (rawinputs['nbtag']>=3) # signal_selection 
    isbackground = ~issignal
    bkgnormed = normedinputs[isbackground]
    bkg = rawinputs[isbackground]
    xmax = np.reshape(inputmeans + 5* inputsigma, inputmeans.shape[1])

    m = ABCDdnn(ncat_per_feature, inputdim, minibatch=minibatch, conddim=conddim, LRrange=LRrange, \
        beta1=beta1, beta2=beta2, nafdim=nafdim, depth=depth, savedir=savedir, retrain=retrain, seed=seed)
    m.setrealdata(bkgnormed)
    m.savehyperparameters()
    m.monitorevery = 100

    if train:
        m.train(steps)
        m.display_training()

    nj9cut = True
    if nj9cut:   
        ncol=3 # for plots below
        condlist = [
            [[1., 0., 0.,   1., 0., ]],
            [[0., 1., 0.,   1., 0., ]],
            [[0., 0., 1.,   1., 0., ]],
            [[1., 0., 0.,   0., 1., ]],
            [[0., 1., 0.,   0., 1., ]],
            [[0., 0., 1.,   0., 1., ]]
        ]
        select0 = (rawinputs['njet']==7) & (rawinputs['nbtag']==2)
        select1 = (rawinputs['njet']==8) & (rawinputs['nbtag']==2)
        select2 = (rawinputs['njet']>=9) & (rawinputs['nbtag']==2)
        select3 = (rawinputs['njet']==7) & (rawinputs['nbtag']>=3)
        select4 = (rawinputs['njet']==8) & (rawinputs['nbtag']>=3)
        select5 = (rawinputs['njet']>=9) & (rawinputs['nbtag']>=3)
        select_data = [select0, select1, select2, select3, select4, select5]

        plottextlist=[
            f'$N_j=7, N_b=2$',
            f'$N_j=8, N_b=2$',
            f'$N_j\geq 9, N_b=2$',
            f'$N_j=7, N_b\geq 3$',
            f'$N_j=8, N_b\geq 3$',
            f'$N_j\geq 9, N_b\geq 3$'
        ]
        njlist = [7, 8, 9, 7, 8, 9]
        nblist = [2, 2, 2, 3, 3, 3]

    else:
        ncol=3 # for plots
        condlist = [
            [[0., 1., 0., 0.,   1., 0., ]],
            [[0., 0., 1., 0.,   1., 0., ]],
            [[0., 0., 0., 1.,   1., 0., ]],
            [[0., 1., 0., 0.,   0., 1., ]],
            [[0., 0., 1., 0.,   0., 1., ]],
            [[0., 0., 0., 1.,   0., 1., ]]
        ]
        select0 = (rawinputs['njet']==8) & (rawinputs['nbtag']==2) 
        select1 = (rawinputs['njet']==9) & (rawinputs['nbtag']==2)
        select2 = (rawinputs['njet']>=10) & (rawinputs['nbtag']==2)
        select3 = (rawinputs['njet']==8) & (rawinputs['nbtag']>=3)
        select4 = (rawinputs['njet']==9) & (rawinputs['nbtag']>=3)
        select5 = (rawinputs['njet']>=10) & (rawinputs['nbtag']>=3)
        select_data = [select0, select1, select2, select3, select4, select5]

        plottextlist=[
            f'$N_j=8, N_b=2$',
            f'$N_j=9, N_b=2$',
            f'$N_j\geq 10, N_b=2$',
            f'$N_j=8, N_b\geq 3$',
            f'$N_j=9, N_b\geq 3$',
            f'$N_j\geq 10, N_b\geq 3$'
        ]

        njlist = [8, 9, 10, 8, 9, 10]
        nblist = [2, 2, 2, 3, 3, 3]

    # create fake data

    fakedatalist = []
    for cond, nj, nb in zip(condlist, njlist, nblist):
        nmcbatches = int(bkgnormed.shape[0] / minibatch)
        nmcremain = bkgnormed.shape[0] % minibatch
        fakelist = []
        cond_to_append = np.repeat(cond, minibatch, axis=0)
        for _ib in range(nmcbatches):
            xin = bkgnormed[_ib*minibatch:(_ib+1)*minibatch, :inputdim]
            xin = np.hstack((xin, cond_to_append)) # append conditional to the feature inputs
            xgen = m.model.predict(xin)
            #xgen = m.generate_sample(cond)
            fakelist.append(xgen)
        # last batch
        xin = bkgnormed[nmcbatches*minibatch:, :inputdim]
        xin = np.hstack((xin, np.repeat(cond, nmcremain, axis=0 ))) # append conditional to the feature inputs
        xgen = m.model.predict(xin)
        fakelist.append(xgen)

        # all data
        fakedata= np.vstack(fakelist)
        fakedata = fakedata * inputsigma[:, :inputdim] + inputmeans[:, :inputdim]
        nfakes = fakedata.shape[0]

        fakedata = np.hstack((fakedata, np.array([nj]*nfakes).reshape((nfakes,1))\
                , np.array([nb]*nfakes).reshape(nfakes,1) )
        )
        fakedatalist.append(fakedata)

    labelsindices = [['MET', 'met', 0.0, xmax[0]], ['H_T', 'ht', 0.0, xmax[1]],\
        ['p_{T5}', 'pt5', 0.0, xmax[2]], ['p_{T6}', 'pt6', 0.0, xmax[3]]]
    nbins=20
    runplots = True
    if runplots:
        yscales = ['log', 'linear']
        for yscale in yscales:
            for li in labelsindices:
                pos = featurevars.index(li[1])
                fig, ax = plt.subplots(2,ncol, figsize=(3*ncol,6))
                iplot = 0
                for fakedata, seld, plottext in zip(fakedatalist, select_data, plottextlist):
                    input_data = rawinputs[seld]
                    # Make ratio plots
                    plotaxes = MplPlotter.ratio_plot(dict(x=input_data[li[1]], bins=nbins, range=(li[2], li[3]), errorbars=True, normed=True, histtype='marker'), \
                        dict(x=fakedata[:, pos], bins=nbins, range=(li[2], li[3]), errorbars=True, normed=True), ratio_range=(0.25, 1.9))
                        
                    plotfig = plotaxes[0][0].get_figure()
                    plotaxes[0][0].set_yscale(yscale)
                    plotfig.set_size_inches(5,5)
                    plotfig.savefig(os.path.join(savedir, f'result_{li[1]}_{iplot}_{yscale}_ratio.pdf'))

                    # make matrix of plots
                    row = iplot // ncol
                    col = iplot % ncol
                    iplot += 1
                    plt.sca(ax[row,col])
                    ax[row,col].set_yscale(yscale)
                    ax[row,col].set_xlabel(f"${li[0]}$ (GeV)")
                    MplPlotter.hist(input_data[li[1]], bins=nbins, alpha=0.5, range=(li[2], li[3]), errorbars=True, histtype='marker', normed=True)
                    MplPlotter.hist(fakedata[:,pos], bins=nbins, alpha=0.5, range=(li[2], li[3]), errorbars=True, normed=True)
                    MplPlotter.hist(bkg[li[1]], bins=nbins, alpha=0.5, range=(li[2], li[3]), histtype='step', normed=True)
                    plt.text(0.6, 0.8, plottext, transform=ax[row,col].transAxes, fontsize=10)
            
                fig.tight_layout()
                fig.savefig(os.path.join(savedir, f'result_matrix_{li[1]}_{yscale}.pdf'))

    generatesigsample = True
    if generatesigsample:
        bkgsigfakedata = np.vstack(fakedatalist)
 
        datadict = {}
        for var, idx in zip(featurevars, range(len(featurevars))):
            datadict[var] = bkgsigfakedata[:, idx]

        writetorootfile(os.path.join(savedir,'fakedata_NAF.root'), datadict)
    pass
