---
title: "Computing Fourier power spectrum"
date: 2024-06-11
---


```python
### Author: Gurprit Singh
### January 5th, 2021
import numpy as np
import sys, getopt
import math
from imageio import imread, imwrite


def vectorized_power_spectrum(X,fourierParams):
    resolution = fourierParams['resolution']
    xfreqStep = fourierParams['xfreqstep']
    yfreqStep = fourierParams['yfreqstep']

    numPoints = X.shape[0]
    xlow = -resolution*0.5 * xfreqStep
    xhigh = resolution*0.5 * xfreqStep
    ylow = -resolution*0.5 * yfreqStep
    yhigh = resolution*0.5 * yfreqStep
    u = np.arange(xlow, xhigh, xfreqStep)
    v = np.arange(ylow, yhigh, yfreqStep)
    uu, vv = np.meshgrid(u,v)
    # print(uu.shape, vv.shape)

    #Compute fourier transform
    dotXU = np.tensordot(X, ([uu,vv]),1)
    angle = 2.0*(math.pi)*dotXU
    realCoeff = np.sum(np.cos(angle),0)
    imagCoeff = np.sum(np.sin(angle),0)

    #return power spectrum
    return (realCoeff**2 + imagCoeff**2) / numPoints


def random_sampler(N):
    sampler = np.zeros((N,2))
    for i in range(N):
        sampler[i][0] = np.random.random_sample()
        sampler[i][1] = np.random.random_sample()
        #print(sampler[i][0],sampler[i][1])
    return sampler

def jitter_sampler(N, scale_domain=1.0):
    sampler = np.zeros((N,2))
    gridsize = int(np.sqrt(N))
    for x in range(gridsize):
        for y in range(gridsize):
            i = x * gridsize + y
            sampler[i][0] = (x + np.random.random_sample())/gridsize
            sampler[i][1] = (y + np.random.random_sample())/gridsize
        #print(sampler[i][0],sampler[i][1])
    sampler *= scale_domain
    return sampler

def readTXT(path):
    return np.loadtxt(open(path, 'rb'), delimiter=' ')


def main(argv):
    outputfile = ''

##############
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ofile="])
    except getopt.GetoptError:
        print('power-spectrum.py -o <outputfile [.exr format]>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('power-spectrum.py -o <outputfile [output.exr]>')
            sys.exit()
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    print ('Output file is "', outputfile)

    SAMPLERNAME='uniform_blue_noise' # jitter # uniform_jitter # blue_noise # uniform_blue_noise
    FOLDERNAME='data/geyser-duration_'+SAMPLERNAME+'/'

    spectralLossParams = {}
    spectralLossParams['resolution'] = 128
    spectralLossParams['xfreqstep'] = 1
    spectralLossParams['yfreqstep'] = 1
    resolution = spectralLossParams['resolution']
    XRES=spectralLossParams['resolution']
    YRES=spectralLossParams['resolution']

    sample_count = 1024
    num_trials = 10
    powerAccum = np.zeros((XRES,YRES))
    for trial in range(num_trials):
        print("trial: ", trial)

        trialstr = str(trial)
        suffix = trialstr.zfill(3)

        ### If you want to test a sampler directly
        samples = jitter_sampler(sample_count,scale_domain=1)

        power = vectorized_power_spectrum(samples,spectralLossParams).astype("float32")

        powerAccum += power
    powerAccum /= num_trials
    powerAccum = powerAccum.astype("float32")
    imwrite(outputfile, powerAccum)



if __name__ == "__main__":
   main(sys.argv[1:])

```



