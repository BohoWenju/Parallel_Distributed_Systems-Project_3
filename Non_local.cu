/*

                      MOURELATOS ELEFTHERIOS
                            AEM 9437

    Implementing the Non_local means filter with the use of gpu.
    Each function is being sent to gpu.





*/






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
void d_fill(float* patches,float* d,int i,int patchsize,float patchsigma,float filtsigma,float* gauss,int n){
  int index = blockIdx.x*blockDim.x+threadIdx.x;
  float sum=0;
  if (index < powf(n,2)){
    if (index==i)
      d[index*n*n+i]=1;
    else{
      for (int j=0; j<patchsize*patchsize; j++)
        sum+=gauss[j]*powf(patches[index*patchsize*patchsize+j]-patches[i*patchsize*patchsize+j],2);
      d[index*n*n+i]=sum;
    }
  }
}

// get the patch of an image in points (x,y)
// with patchsize
__global__
void patch(float* B,float* patches,int i,int j,int size,int patchsize,int n){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int h_patch=(patchsize-1)/2;
  int x=index/size;
  int y=index%size;
    if ((x<size)&&(y<size)&&(x+h_patch<n)&&(y+h_patch<n)){
      int patch_index=x*n*patchsize*patchsize+y*patchsize*patchsize+i*patchsize+j;
      patches[patch_index]=B[(x-h_patch+j)*size+(y-h_patch+i)];
  }
}


// function to mirror the array
__global__
void mirroring(float* A,float* B,int patchsize,int n){
  int h_patch=(patchsize-1)/2;
  int size=n+2*h_patch;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int x= index/size;
  int y= index%size;
  if (index<(size*size))
    if ((x<h_patch)||(x>n)||(y<h_patch)||(y>n))
      B[index]=0;
    else
      B[index]=A[index];
}

__global__
void matrix_vec(float* d,float* vec,float* out,int n){
  int index= blockIdx.x*blockDim.x+threadIdx.x;
  int x= index/n;
  int y= index%n;
  float sum=0;
  float sum1=0;
  if (index<(n*n)){
    for(int i=0; i<(n*n); i++){
      sum+=d[x*n*n*n*n+y*n*n+i]*vec[i];
      sum1+=d[index*n*n+i];
    }
    out[index]=sum/sum1;
  }
}

void filter(float** A,int n,int patchsize,float patchsigma,float filtsigma){

  // generating image
  float* temp_out=(float*)malloc(n*n*sizeof(float));
  for(int i=0; i<n; i++)
    for (int j=0; j<n; j++)
      temp_out[i*n+j]=A[i][j];

  // mirroring array A
  int h_patch=(patchsize-1)/2;
  int size=n+2*h_patch;

  // CUDA mirroring
  float *dev_b,*dev_a;
  cudaMalloc((void**)&dev_a,n*n*sizeof(float));
  cudaMalloc((void**)&dev_b,size*size*sizeof(float));
  cudaMemcpy(dev_a,temp_out,n*n*sizeof(float),cudaMemcpyHostToDevice);
  mirroring<<<(size*size)/NTHREADS_PER_BLOCK,NTHREADS_PER_BLOCK>>>(dev_a,dev_b,patchsize,n);


  float* dev_patches;
  cudaMalloc((void**)&dev_patches,(n*n)*(patchsize*patchsize)*sizeof(float));
  for (int i=0; i<patchsize; i++)
    for (int j=0; j<patchsize; j++)
      patch<<<(n*n)/NTHREADS_PER_BLOCK,NTHREADS_PER_BLOCK>>>(dev_b,dev_patches,i,j,size,patchsize,n);

  cudaFree(dev_b);


  // calculate gaussian weight
  // calculate distance for each patch
  float* gauss=(float*)malloc(patchsize*patchsize*sizeof(float));
  float sum=0;
  for (int i=0; i<patchsize*patchsize; i++){
    int x=i%(patchsize);
    int y=i/(patchsize);
    x=x-((patchsize)-1)/2;
    y=y-((patchsize)-1)/2;
    gauss[i]= (1/(2.0*patchsigma))*exp(-(x*x+y*y)/(2*M_PI*patchsigma));
    sum+=(1/(2.0*patchsigma))*exp(-(x*x+y*y)/(2*M_PI*patchsigma));
  }
  for (int i=0; i<patchsize*patchsize; i++)
    gauss[i]/=sum;

  float* dev_d,*dev_gauss;
  cudaMalloc((void**)&dev_gauss,pow(patchsize,2)*sizeof(float));
  cudaMalloc((void**)&dev_d,pow(n,4)*sizeof(float));
  cudaMemcpy(dev_gauss,gauss,pow(patchsize,2)*sizeof(float),cudaMemcpyHostToDevice);
    for (int i=0; i<(n*n); i++)
        d_fill<<<(n*n)/NTHREADS_PER_BLOCK,NTHREADS_PER_BLOCK>>>(dev_patches,dev_d,i,patchsize,patchsigma,filtsigma,dev_gauss,n);

  cudaFree(dev_patches);
  cudaFree(dev_gauss);


  float* dev_out;
  cudaMalloc((void**)&dev_out,n*n*sizeof(float));
  matrix_vec<<<(n*n)/NTHREADS_PER_BLOCK,NTHREADS_PER_BLOCK>>>(dev_d,dev_a,dev_out,n);

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

  //  Translating INPUT


  if (argc<=1){
    printf("\nNo image was passed as input,exiting...\n");
    exit(1);
  }
  if (argc <=2){
    printf("\nNo image dimensions (dim1xdim2) were parsed,exiting...\n");
    exit(1);
  }

  char* img_name;
  img_name=(char*)malloc(strlen(argv[1])*sizeof(char));
  if (img_name==NULL){
    printf("\nNo memory can be allocated,exiting...\n");
    exit(-1);
  }
  memcpy(img_name,argv[1],strlen(argv[1]));

  // getiing dimensions of image from input
  //int rows_t,cols_t;
  bool flag=true;
  char *rows_str,*cols_str;
  int size_r=0;
  int size=strlen(argv[2]);
  for (int i=0; i<strlen(argv[2]); i++){
    if (((argv[2][i]=='X')||(argv[2][i]=='x'))&&(flag)){
      rows_str=(char*)malloc((i-1)*sizeof(char));
      for (int j=0; j<i; j++)
        rows_str[j]=argv[2][j];
      flag=false;
      size_r=i+1;
    }
    if ((!flag)&&(i==(strlen(argv[2])-1))){
      cols_str=(char*)malloc((i-size_r+1)*sizeof(char));
      for (int j=size_r; j<=i; j++)
        cols_str[j-size_r]=argv[2][j];
    }
  }
  int rows=atoi(rows_str);
  int cols=atoi(cols_str);
  free(rows_str);
  free(cols_str);


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
  fp=fopen(img_name,"r");
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

  // creating names for the new files:
  // image_name+"_noised.txt"
  // image_name+_filtered.txt"
  // image_name+_residual.txt
  // to store the output

/*
  for (int i=0; i<4; i++)
    img_name[strlen(img_name)-1]='\0';

  char* n_img_name=(char*)malloc(strlen(img_name)*sizeof(char));
  if (n_img_name==NULL){
    printf("\nNo memory can be allocated,exiting...\n");
    exit(-1);
  }
  memcpy(n_img_name,img_name,strlen(img_name));
  strcat(n_img_name,"_noised.txt");

  char* img_filt=(char*)malloc(strlen(img_name)*sizeof(char));
  if (img_filt==NULL){
    printf("\nNo memory can be allocated,exiting...\n");
    exit(-1);
  }
  memcpy(img_filt,img_name,strlen(img_name));
  strcat(img_filt,"_filtered.txt");

  char* img_res=(char*)malloc(strlen(img_name)*sizeof(char));
  if (img_res==NULL){img_filt
    printf("\nNo memory can be allocated,exiting...\n");
    exit(-1);
  }
  memcpy(img_res,img_name,strlen(img_name));
  strcat(n_img_name,"_residual.txt");

*/
  free(img_name);


  //*********************************************

  //                  PARAMETERS

  int patchsize=5;
  float noise_sigma=0.01;
  float patchsigma=0.01;
  float filtsigma=1;

  //*********************************************


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

  // freeing memory from original array since it's no longer needed
  for (int i=0; i<rows; i++)
    free(img_Arr[i]);
  free(img_Arr);

  //*******************************************




  //*******************************************

  //          Patch/Filter implementation

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

  clock_gettime(CLOCK_MONOTONIC,&ts_end);
  printf("\nDone!\nTime for filter: %lf \n",( (double)ts_end.tv_sec +(double)ts_end.tv_nsec*pow(10,-9)- (double)ts_start.tv_sec -(double)ts_start.tv_nsec*pow(10,-9)));

  //          Creating Filtered image
  FILE* f_filter=fopen("filtered.txt","w");
  if (f_filter==NULL){
    printf("\nCouldn't open filtered image file,exiting...\n");
    exit(-1);
  }
  for (int i=0; i<rows; i++)
    for (int j=0; j<cols; j++)
      fprintf(f_filter,"%f",n_img_Arr[i][j]);
  fclose(f_filter);

  //         Creating Residual image
  FILE* f_ref=fopen("residual.txt","w");
  for (int i=0; i<rows; i++)
    for (int j=0; j<cols; j++)
      fprintf(f_ref,"%f",fabs(n_img_Arr[i][j]-residual[i][j]));
  fclose(f_ref);


  //*******************************************
  return 0;
}
