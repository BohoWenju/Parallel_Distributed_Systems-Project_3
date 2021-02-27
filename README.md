# Parallel_Distributed_Systems-Project_3

In this project we were asked to implement the Non_local means algorithm both in serial version and in parallel.
Parallel implementation was done by taking advantage of the GPU,and it's shared memory property.
Images must be passed as arguments along with their dimensions(must be symmetric) in .txt format.
i.e.:
    ./script_name img_name.txt dim1xdim1
    
 To help with the conversion of an image to .txt a matlab script was created called image_convert_2_txt.m .
 In order to view the resulting image you can take advantage of the show_image.m natlab script.
 
 
