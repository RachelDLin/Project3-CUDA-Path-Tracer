# CUDA Path Tracer

Rachel Lin

* [LinkedIn](https://www.linkedin.com/in/rachel-lin-452834213/)
* [personal website](https://www.artstation.com/rachellin4)
* [Instagram](https://www.instagram.com/lotus_crescent/)



Tested on: Windows 11, 12th Gen Intel(R) Core(TM) i7-12700H @ 2.30GHz, NVIDIA GeForce RTX 3080 Laptop GPU (16 GB)





## Overview

This is a basic Monte Carlo pathtracer implemented on the GPU using CUDA kernels. A ray is cast from each pixel on the screen, and is assigned a computation thread. Each "bounce" the pathtracer checks for intersections with the scene. Every time an intersection is detected, the ray either reflects (or refracts) into a new ray or gets terminated, and color is accumulated based on the material properties of the surface intersected.



Additional support is provided for refractive, specular, and diffuse shaders, as well as several common post-process filters and effects. Toggles are also provided for several optimization techniques to enhance performance.



A scene is represented in a JSON file containing information on the materials, mesh objects, and camera. The camera object also contains settings for the number of frames to iterate over and the maximum number of times a ray an bounce. You will find a few example scenes under the "scenes" folder.



## Features

* Basic pathtracing logic for diffuse shading
* Work-efficient stream compaction on rays 2
* Sorting rays by material
* Russian roulette path termination 1
* Stochastic sampled antialiasing
* Refractive materials 2
* Microfacet roughness
* Physically-based depth of field 2
* TODO: Mesh loading (obj, gltf) 6
* TODO: BVH acceleration
* TODO: Texture mapping 6
* TODO: Bump mapping
* TODO: Post-process bloom 3
* TODO: Post-process edge detection
* TODO: Post-process quantization
* TODO: Post-process pixelation
* TODO: Post-process anisotropic kuwahara filter



## Shading

### Diffuse Lighting





### Refraction





### Reflection/Transmission







## File Loading

### Custom Meshes





#### External Libraries

* [tiny\_obj](https://github.com/syoyo/tinyobjloader)
* [tiny\_gltf](https://github.com/syoyo/tinygltf/)





### Custom Textures







## Filters \& Post Processing

### Stochastic Sampled Antialiasing





### Physically-Based Depth of Field





### Convolution Bloom





### Sobel Edge Detection





### Quantization





### Pixelation (Point Filter)





### Anisotropic Kuwahara







## Optimizations

### Memory Coalescing





### Work-Efficient Stream Compaction





### Russian Roulette Path Termination





### BVH Acceleration

