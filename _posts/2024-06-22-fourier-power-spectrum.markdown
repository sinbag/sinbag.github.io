---
layout: post
title:  "Fourier power spectrum "
date:   2024-06-22 19:01:00 +0200
# categories: jekyll update
---


## How to compute the power spectrum of point samples

### Generating point samples
Let's first generate some point samples 

```python
# Generate jittered samples in 2D
def jitter_sampler(N):
    sampler = np.zeros((N,2))
    gridsize = int(np.sqrt(N))
    for x in range(gridsize):
        for y in range(gridsize):
            i = x * gridsize + y
            sampler[i][0] = (x + np.random.random_sample())/gridsize
            sampler[i][1] = (y + np.random.random_sample())/gridsize
    return sampler
```

### Computing the power spectrum (slow version)

Assuming we have `N` point samples stored in an array named `samples`, the code to compute the power spectrum is quite simple:

```python
# xres: spectrum width, yres: spectrum height
def power_spectrum(samples, xres, yres):
    # Initialize a 2D array to store the power 
    power = np.zeros((xres,yres))
    power = power.astype("float32")
    
    # Define some variables 
    half_xres = int(xres * 0.5)
    half_yres = int(yres * 0.5)

    # number of point samples
    N = samples.shape[0]

    xindex = 0
    # loop from -ve to +ve frequencies; 
    # this is important to ensure the DC component is at the center of the image
    for xfreq in range(-half_xres,half_xres,1):
        yindex = 0
        for yfreq in range(-half_yres,half_yres,1):
            real = 0
            imag = 0
            # This is the main loop, computing real and imaginary Fourier coefficients
            # For each frequency coordinate (xfreq, yfreq), we go over all the points
            for i in range(N):
                expterm = 2*np.pi*((samples[i][0]*(xfreq)) +  (samples[i][1]*(yfreq)))
                real += np.cos(expterm)
                imag += np.sin(expterm)
            # The power is given by: real^2 + imag^2
            power[xindex,yindex] = (real*real + imag*imag) / N
            yindex += 1
        xindex += 1
    # returns the power
    return power
```


### Computing the power spectrum in vectorized form (fast version)

```python 
# Vectorized version of the above code; extremely fast
def vectorized_power_spectrum(X, xres, yres):
    N = X.shape[0]
    xlow = -xres*0.5
    xhigh = xres*0.5
    ylow = -yres*0.5
    yhigh = yres*0.5
    u = np.arange(xlow, xhigh, 1.0)
    v = np.arange(ylow, yhigh, 1.0)
    uu, vv = np.meshgrid(u,v)

    #Compute fourier transform
    dotXU = np.tensordot(X, ([uu,vv]),1)
    angle = 2.0*(np.pi)*dotXU
    realCoeff = np.sum(np.cos(angle),0)
    imagCoeff = np.sum(np.sin(angle),0)

    #return power spectrum
    return (realCoeff**2 + imagCoeff**2) / N


```


### Storing the output power spectrum in an HDR file (.exr format)

```python
import OpenEXR
import Imath

# Function to write EXR image
def write_exr_gray(file_path, image):
    height, width = image.shape

    # Convert to OpenEXR format
    exr_image = OpenEXR.OutputFile(file_path, OpenEXR.Header(width, height))

    # one channel array
    red = image[:, :].astype(np.float32).tobytes()
    green = image[:, :].astype(np.float32).tobytes()
    blue = image[:, :].astype(np.float32).tobytes()
    
    # write the one channel image
    exr_image.writePixels({'R': red, 'G': green, 'B': blue})
```



### Example usage

```python
if __name__ == "__main__":

    N = 4096 # number of samples 

    # Generate samples
    point_samples = jitter_sampler(N)

    # Declare the resolution of the power spectrum image
    height = 64
    width = 64

    # Compute the power specturm
    # power = power_spectrum(point_samples, width, height)
    power = vectorized_power_spectrum(point_samples, width, height)

    # Write power as EXR image
    write_exr_gray("./power.exr", power)
```