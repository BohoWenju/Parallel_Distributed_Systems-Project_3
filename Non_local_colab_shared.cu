#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define NTHREADS_PER_BLOCK 256
#define BLOCKS 64

float inverse_gausian(float median,float sigma,float random_value){
  float y1=median + sqrt(-2*pow(sigma,2)*log(sqrt(2*M_PI)*sigma*random_value));
  float y2=median - sqrt(-2*pow(sigma,2)*log(sqrt(2*M_PI)*sigma*random_value));
  float x=(float)rand()/RAND_MAX;
  return (x<0.5)?y1:y2;
}

void apply_Noise(int rows,int cols,float** In,float** Out,float sigma){
  for (int i=0; i<rows; i++)
    for (int j=0; j<cols; j++){
      double z=inverse_gausian(In[i][j],sigma,(float)rand()/RAND_MAX);
      if (z > 1)
        Out[i][j]=1;
      else if (z < 0)
        Out[i][j]=0;
      else
        Out[i][j]=z;
    }
}

__global__
void d_fill(float* B,float* d,float* patches,int i,int patchsize,float* gauss,int n,int size_patches,float filtsigma){

  // each block has i.e. p^2 elements starting from the top row
  // therefore number of cols of specific block is n
  // and blockdim = rows*n ==> rows=(blockdim/n)
  // filling the array of patches is like a 2d stencil problem
  // passing the patches of one block in a shared memory
  int h_patch=(patchsize-1)/2;
  int size_rows=n+2*h_patch; // basically number of columns
  extern __shared__ float sh_patches[];
  int index = blockIdx.x*blockDim.x+threadIdx.x;

  // adding h_patch in order to navigate through the B array
  // basically copying a part of B array to shared memory
  int x=index/n+h_patch;
  int y=index%n+h_patch;
  int l_x=threadIdx.x/n;
  int l_y=threadIdx.x%n;
  if (index<(n*n)){
    sh_patches[(l_x+h_patch)*size_rows+(l_y+h_patch)]=B[x*size_rows+y];
    if ((l_y<h_patch)&&(l_x<h_patch)){
      sh_patches[l_x*size_rows+l_y]=B[(x-h_patch)*size_rows+(y-h_patch)];
      sh_patches[(l_x+h_patch+blockDim.x/n)*size_rows+(l_y+h_patch+n)]=B[(x+n)*size_rows+(y+n)];
    }
    else if (l_y<h_patch){
      sh_patches[(l_x+h_patch)*size_rows+l_y]=B[x*size_rows+(y-h_patch)];
      sh_patches[(l_x+h_patch)*size_rows+(l_y+h_patch+n)]=B[x*size_rows+(y+n)];
    }
    else if ((l_x<h_patch)){
      sh_patches[(l_x)*size_rows+(l_y+h_patch)]=B[(x-h_patch)*size_rows+y];
      sh_patches[(l_x+h_patch+blockDim.x/n)*size_rows+(l_y+h_patch)]=B[(x+n)*size_rows+y];
    }
  }

  __syncthreads();

  // d_filling the array by taking advantage the shared memory above
  // each element is (x+blockIdx.x),(y+blockIdx.x)
  // if i belongs to the same block as our thread then both readings
  // come from shared memory
  // else the reading of element i comes from global


  int f_index=(l_x)*n*patchsize*patchsize+(l_y)*patchsize*patchsize;
  // normilizing our data
  x=i/n;
  y=i%n;
  float sum=0;
  // in order to check whether the element comes from the same block
  if ((i/blockDim.x)==(blockIdx.x)){
    int n_i=i%blockDim.x;
    x=n_i/n;
    y=n_i%n;
    int s_index=x*n*patchsize*patchsize+y*patchsize*patchsize;
    if (index < powf(n,2)){
      if (index==n_i)
        d[index*n*n+i]=1;
      else {
        for (int j=0; j<patchsize; j++)
          for (int k=0; k<patchsize; k++)
            sum+=gauss[j]*powf(sh_patches[f_index+k*patchsize+j]-sh_patches[s_index+k*patchsize+j],2);
        d[index*n*n+i]=expf(-sum/powf(filtsigma,2));
        }
      }
    }
    if (index==i)
      d[index*n*n+i]=1;
    else{
      x=i/n;
      y=i%n;
      int s_index=x*n*patchsize*patchsize+y*patchsize*patchsize;
      for (int j=0; j<patchsize; j++)
        for (int k=0; k<patchsize; k++)
          sum+=gauss[j]*powf(sh_patches[f_index+k*patchsize+j]-patches[s_index+k*patchsize+j],2);
      d[index*n*n+i]=expf(-sum/powf(filtsigma,2));
    }
}

// get the patch of an image in points (x,y)
// with patchsize
__global__
void patch(float* B,float* patches,int i,int j,int size,int patchsize,int n){
  int index = blockIdx.x * NTHREADS_PER_BLOCK + threadIdx.x;
  int h_patch=(patchsize-1)/2;
  int x=index/size;
  int y=index%size;
    if ((x<size)&&(y<size)){
        if ((y>=h_patch)&&(y<n+h_patch)&&(x<n+h_patch)&&(x>=h_patch)){
            int patch_index=(x-h_patch)*n*patchsize*patchsize+(y-h_patch)*patchsize*patchsize+j*patchsize+i;
            patches[patch_index]=B[(x+i-h_patch)*size+(y+j-h_patch)];
    }
  }
}


// function to mirror the array
__global__
void mirroring(float* A,float* B,int patchsize,int n){
  int h_patch=(patchsize-1)/2;
  int size=n+2*h_patch;
  int index = blockIdx.x * NTHREADS_PER_BLOCK + threadIdx.x;
  int x= index/size; // row
  int y= index%size; // col
  if (index<(size*size))
    if ((x<h_patch)||(x>=n+h_patch)||(y<h_patch)||(y>=n+h_patch))
      B[index]=0;
    else
      B[index]=A[(x-h_patch)*n+(y-h_patch)];
}

__global__
void matrix_vec(float* d,float* vec,float* out,int n){
  int index= blockIdx.x*blockDim.x+threadIdx.x;
  int x=index/n;
  int y=index%n;
  int d_x=x*powf(n,3);
  int d_y=y*powf(n,2);
  float sum=0;
  float sum1=0;
  if (index<(n*n)){
    for(int i=0; i<(n*n); i++){
      sum+=d[d_x+d_y+i]*vec[i];
      sum1+=d[d_x+d_y+i];
    }
    out[index]=sum/sum1;
  }
}

void filter(float** A,int n,int patchsize,float patchsigma,float filtsigma){


  int h_patch=(patchsize-1)/2;
  int size=n+2*h_patch;

  // generating image as 1d array
  float* temp_out=(float*)malloc(n*n*sizeof(float));
  for(int i=0; i<n; i++)
    for (int j=0; j<n; j++)
      temp_out[i*n+j]=A[i][j];

  // calculate gaussian weight
  // calculate distance for each patch
  float* gauss=(float*)malloc(patchsize*patchsize*sizeof(float));
  float sum=0;
  for (int i=0; i<patchsize*patchsize; i++){
    int x=i/(patchsize);
    int y=i%(patchsize);
    x=x-((patchsize)-1)/2;
    y=y-((patchsize)-1)/2;
    gauss[i]= (1/(2.0*patchsigma))*exp(-(x*x+y*y)/(2*M_PI*patchsigma));
    sum+=(1/(2.0*patchsigma))*exp(-(x*x+y*y)/(2*M_PI*patchsigma));
  }
  for (int i=0; i<patchsize*patchsize; i++)
    gauss[i]/=sum;

  // CUDA mirroring, the allocation of variables dev is not being done
  // immediately
  float *dev_b,*dev_a;
  cudaMalloc((void**)&dev_a,n*n*sizeof(float));
  cudaMalloc((void**)&dev_b,size*size*sizeof(float));
  cudaMemcpy(dev_a,temp_out,n*n*sizeof(float),cudaMemcpyHostToDevice);
  // size*size+NTHREADS_PER_BLOCK-1/NTHREADS_PER_BLOCK because if
  // size*size is < NTHREADS_PER_BLOCK integer division will give 0
  mirroring<<<(size*size+NTHREADS_PER_BLOCK-1)/NTHREADS_PER_BLOCK,NTHREADS_PER_BLOCK>>>(dev_a,dev_b,patchsize,n);


  float* dev_patches;
  cudaMalloc((void**)&dev_patches,(n*n)*(patchsize*patchsize)*sizeof(float));
  for (int i=0; i<patchsize; i++)
    for (int j=0; j<patchsize; j++)
      patch<<<(size*size+NTHREADS_PER_BLOCK-1)/NTHREADS_PER_BLOCK,NTHREADS_PER_BLOCK>>>(dev_b,dev_patches,i,j,size,patchsize,n);

  float* dev_d,*dev_gauss;
  cudaMalloc((void**)&dev_gauss,pow(patchsize,2)*sizeof(float));
  cudaMalloc((void**)&dev_d,pow(n,4)*sizeof(float));
  cudaMemcpy(dev_gauss,gauss,pow(patchsize,2)*sizeof(float),cudaMemcpyHostToDevice);

  // shared memory needs to be sizeof block +
  int shared_arr=(n*n)/BLOCKS;
  int r_shared_arr=shared_arr/n+2*h_patch;
  size_t size_patches=(r_shared_arr*r_shared_arr)*sizeof(float);
    for (int i=0; i<(n*n); i++)
        d_fill<<<(n*n+NTHREADS_PER_BLOCK-1)/NTHREADS_PER_BLOCK,NTHREADS_PER_BLOCK,size_patches>>>(dev_b,dev_d,dev_patches,i,patchsize,dev_gauss,n,size_patches,filtsigma);

  cudaFree(dev_patches);
  cudaFree(dev_b);
  cudaFree(dev_gauss);


  float* dev_out;
  cudaMalloc((void**)&dev_out,n*n*sizeof(float));
  matrix_vec<<<(n*n+NTHREADS_PER_BLOCK-1)/NTHREADS_PER_BLOCK,NTHREADS_PER_BLOCK>>>(dev_d,dev_a,dev_out,n);

  cudaFree(dev_a);
  cudaFree(dev_d);
  cudaMemcpy(temp_out,dev_out,n*n*sizeof(float),cudaMemcpyDeviceToHost);
  cudaFree(dev_out);
  for (int i=0; i<n; i++)
    for (int j=0; j<n; j++)
      A[j][i]=temp_out[j*n+i];

  free(temp_out);
}





int main(int argc,char* argv[]){

  int rows=64;
  int cols=64;

  float** img_Arr=(float**)malloc(rows*(sizeof(float*)));
  if (img_Arr==NULL){
    printf("\nNo memory can be allocated,exiting...\n");
    exit(-1);
  }
  for (int i=0; i<rows; i++){
    img_Arr[i]=(float*)malloc(cols*sizeof(float));
    if (img_Arr[i]==NULL){
      printf("\nNo memory can be allocated,exiting...\n");
      exit(-1);
    }
  }

  printf("Reading image...\n");
  FILE* fp;
  fp=fopen("./new_image.txt","r");
  if (fp==NULL){
    printf("\nThere is no such file,exiting...\n");
    exit(1);
  }

  float num;
  for (int i=0; i<rows; i++)
    for(int j=0; j<cols; j++){
      fscanf(fp,"%f",&num);
      if (num==EOF)
        break;
      img_Arr[i][j]=num;
    }
  fclose(fp);

  printf("\nDone Reading!\nThe rows are: %d and the columns are: %d\n",rows,cols);

  // NORMILIZING IMAGE
  printf("Normilizing Image...\n\n");
  float min=0;
  float max=0;
  for (int i=0; i<rows; i++)
    for (int j=0; j<cols; j++)
      if (min>img_Arr[i][j])
        min = img_Arr[i][j];

  for (int i=0; i<rows; i++)
    for (int j=0; j<cols; j++)
      if (max<(img_Arr[i][j]-min))
        max= img_Arr[i][j]-min;

  for (int i=0; i<rows; i++)
    for (int j=0; j<cols; j++)
        img_Arr[i][j]= (img_Arr[i][j]-min)/max;



  struct timespec ts_start,ts_end;
// image_Arr now holds the pixel values of the image

// applying gaussian noise to image
// creating a new file with name image_name+"_noised.txt"
// to store the output

  //                  PARAMETERS
  int patchsize=5;
  float noise_sigma=0.001;
  float patchsigma=5/3;
  float filtsigma=0.02;

  //*********************************************

  //             NOISE PARSING
  float** n_img_Arr=(float**)malloc(rows*sizeof(float*));
  for (int i=0; i<rows; i++)
    n_img_Arr[i]=(float*)malloc(cols*sizeof(float));

  clock_gettime(CLOCK_MONOTONIC,&ts_start);

  // aplying noise cannot be parellized due to the random nature
  // of the algorithm
  apply_Noise(rows,cols,img_Arr,n_img_Arr,noise_sigma);

  //          Manipulating File
  FILE* f_noise=fopen("noised.txt","w");
  if (f_noise==NULL){
    printf("\nCouldn't open noised image file,exiting...\n");
    exit(-1);
  }
  for (int i=0; i<rows; i++)
    for (int j=0; j<cols; j++)
      fprintf(f_noise,"%f",n_img_Arr[i][j]);
  fclose(f_noise);
  clock_gettime(CLOCK_MONOTONIC,&ts_end);
  printf("\nDone!\nTime for noise parsing: %lf \n",( (double)ts_end.tv_sec +(double)ts_end.tv_nsec*pow(10,-9)- (double)ts_start.tv_sec -(double)ts_start.tv_nsec*pow(10,-9)));


  //*******************************************

  // freeing memory from original array since it's no longer needed
  for (int i=0; i<rows; i++)
    free(img_Arr[i]);
  free(img_Arr);


  //*******************************************

  //          patch/Filter implementation


  //          Initializing residual
  float** residual=(float**)malloc(rows*sizeof(float*));
  for (int i=0; i<cols; i++)
    residual[i]=(float*)malloc(cols*sizeof(float));
  for(int i=0; i<rows; i++)
    for(int j=0; j<cols; j++)
      residual[i][j]=n_img_Arr[i][j];



  printf("\nApplying Filter...\n");
  clock_gettime(CLOCK_MONOTONIC,&ts_start);

  filter(n_img_Arr,rows,patchsize,patchsigma,filtsigma);

  //          Manipulating File
  FILE* f_filter=fopen("filtered.txt","w");
  if (f_filter==NULL){
    printf("\nCouldn't open filtered image file,exiting...\n");
    exit(-1);
  }
  for (int i=0; i<rows; i++)
    for (int j=0; j<cols; j++)
      fprintf(f_filter,"%f",n_img_Arr[i][j]);
  fclose(f_filter);
  clock_gettime(CLOCK_MONOTONIC,&ts_end);
  printf("\nDone!\nTime for filter: %lf \n",( (double)ts_end.tv_sec +(double)ts_end.tv_nsec*pow(10,-9)- (double)ts_start.tv_sec -(double)ts_start.tv_nsec*pow(10,-9)));

  //         Creating Residual
  FILE* f_ref=fopen("residual.txt","w");
  for (int i=0; i<rows; i++)
    for (int j=0; j<cols; j++)
      fprintf(f_ref,"%f",fabs(n_img_Arr[i][j]-residual[i][j]));
  fclose(f_ref);


  //*******************************************
  return 0;
}
