---
layout: research-note
title: "Neural materials basics (Part 1)"
date:   2025-12-24 11:59:00 +0200
author: Gurprit Singh
series: Neural materials
math: true
permalink: /writing/neural-materials-part-1/
next_note_url: /writing/neural-materials-part-2/
next_note_title: Neural materials basics (Part 2)
---


### Learn how to load textures

### How to encode textures in an MLP?

### How positional encoding helps?

### What are mipmaps and how to encode them in an MLP?

### What are materials?

### How materials are encoded in an MLP using textures (albedo, etc.)?

We pass (u,v) or pixel coordinates of textures as input and let the network predict all the channels (albedo, diffuse, specular, roughness).
During inference, for each coordinate the network can predict all the channels. 
During shading, we use these predicted textures and pass it to our shading function (Lambertian, Blinn-Phong, GGX). 
We can also play with the lighting, and train a relightable neural material.

### What are scale-conditioned material encoding? 

To acucrately predict neural materials at different zoom (scale/distance from screen), we
- first generate mipmap levels of all channels (ao, diffuse, specular, normal, displacement, etc.)
- Train a model that takes as input both the UV coordinates and the input mipmap (scale) value.
- During training, we randomly select a mipmap level (scale) and create a batch of UV coordinates at that level, and compute the loss.
- At each iteration, a different scale is randomly picked and the network tries to learn its value
- Once fully trained, this network can predict channels at any scale

Question: Can we predict the materials at an arbitrary scale value? Given that we have trained the scale-conditioned material MLP at 
only fewer discrete mipmap levels.

#### What are the common glitches in the code?

It is common error to flip the U,V coordinates while sampling the texture at a given scale.

