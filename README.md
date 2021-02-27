# Parallel_Distributed_Systems-Project_3

In this project we were asked to implement the Non_local means algorithm both in serial version and in parallel.
Parallel implementation was done by taking advantage of the GPU,and it's shared memory property.
Images must be passed as arguments along with their dimensions(must be symmetric) in .txt format.
i.e. :
    ./script_name img_name.txt dim1xdim1
    
 To help with the conversion of an image to .txt a matlab script was created called image_convert_2_txt.m .
 In order to view the resulting image you can take advantage of the show_image.m natlab script.
 
 # Implementations
 
 **Non_local.c** : In this programm the Non_local means algorithm that was given to us is being implemented in
 basic c.An image is being passed as input and and gaussian noise is being parsed to it.In order to clear the noise the 
 filter function implements the Non_local means algorithm to the image.The output of the programm is the time interval 
 between noise parsing and filter usage.Finally,three new files are created called "noised.txt","filtered.txt" and "residual.txt"
 where residual holds the noised image's pixels after substracting the filtered image's pixels.
 
 **Non_local.cu** : In this programm the above algorithm is being executed in parallel.Every function is being converted
 to it's parallelized version in cuda.Note that,the apply Noise function can't be implemented to Cuda due to it's randomness.
 
 **Non_local_shared.cu** : For the final implementation of the Non_local means algorithm i am taking advantage of the shared memory 
 of each block in the GPU.During the calculation of the d_array (where d hold the term : exp(-dist/pow(filtsigma,2)) ), the expanded image 
 in being chopped in blocks and my following a logic,kinda like a 2d stencil, the patches of the elements of the specific block are 
 stored in shared memory.
 
 
 
 
 
