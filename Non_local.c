/*

                      MOURELATOS ELEFTHERIOS
                            AEM 9437

            Implementing the Non_local means filter.

    Applying the formula that was being given to us,as well as i
    understood it.Mirroring is basically an expanding of the original
    array by 2*h_patch(h_patch=(patchsize-1)/2) with 0s.
    The input must be in the form :
    - script_name img_name.txt dim1xdim1
    (image must be symmetric)
    The output of this program is being stored in .txt files with
    names:
    - img_name_noised.txt
    - img_name_filtered.txt
    - img_name_residual.txt

    By using the matlab script(show_image.m) the image represented by
    those files will be viewed.



*/



#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <math.h>


// generate random variables that follow gaussian distribution
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

float dist_calc_g(float* vector1,float* vector2,int patchsize,float patchsigma){
  float dist=0;
  float patch_sq=patchsigma*patchsigma;
  //float sum=0;
  for (int i=0; i<patchsize*patchsize; i++){
    int x=i%(patchsize);
    int y=i/(patchsize);
    x=x-((patchsize)-1)/2;
    y=y-((patchsize)-1)/2;
    float ga= (1/(2.0))*exp(-(x*x+y*y)/(2*M_PI*patch_sq));
    dist+=ga*pow(vector1[i]-vector2[i],2);
    //sum+=ga;
  }
  return dist;
}

// get the patch of an image in points (x,y)
// with patchsize
float* patch(float** A,int x,int y,int patchsize){
  float* patch=(float*)malloc(patchsize*patchsize*sizeof(float));
  if (patch==NULL){
    printf("No memory can be allocated,exiting...\n");
    exit(-1);
  }
  int temp=0;
  int h_patch=(patchsize-1)/2;
  for (int i=0; i<patchsize; i++)
    for (int j=0; j<patchsize; j++){
      patch[temp]=A[(x-h_patch)+j][(y-h_patch)+i];
      temp++;
    }

  return patch;
}


// function to mirror the array
float** mirroring(float** A,int patchsize,int n){
  int h_patch=(patchsize-1)/2;
  int size=n+2*h_patch;
  float** Out=(float**)malloc(size*sizeof(float*));
  if (Out==NULL){
    printf("No memory can be allocated,exiting...\n");
    exit(-1);
  }
  for (int i=0; i<size; i++){
    Out[i]=(float*)malloc(size*sizeof(float));
    if (Out[i]==NULL){
      printf("No memory can be allocated,exiting...\n");
      exit(-1);
    }
  }
  for (int i=0; i<size; i++)
    for (int j=0; j<size; j++)
      if ((i<h_patch)||(i>=n+h_patch)||(j<h_patch)||(j>=n+h_patch))
        Out[i][j]=0;
      else
        Out[i][j]=A[i-h_patch][j-h_patch];


  // enlarging/mirroring the image,not on the corners though
  /*for (int i=0; i<n; i++)
    for (int j=0; j<h_patch; j++){
      Out[j][i+h_patch]=A[h_patch-j][i];
      Out[size-1-j][i+h_patch]=A[(n-1-h_patch)+j][i];
      Out[i+h_patch][j]=A[i][h_patch-j];
      Out[i+h_patch][size-1-j]=A[i][(n-1-h_patch)+j];
    }
  // mirroring the corners of the image
  float** temp_arr=(float**)malloc(h_patch*sizeof(float*));
  for (int i=0; i<h_patch; i++)
    temp_arr[i]=(float*)malloc(h_patch*sizeof(float));

  enum corner{UL,UR,DL,DR};
  for (int i=UL; i<DR; i++){
    if (patchsize==3){
      Out[0][0]=A[1][1];
      Out[size-1][0]=A[n-2][1];
      Out[size-1][size-1]=A[n-2][n-2];
      Out[0][size-1]=A[1][n-2];
      break;
    }

    switch(i){
      case (UL):
        //temp_patch=patch(A,h_patch,h_patch,patchsize);

        for (int i=0; i<h_patch; i++)
          for(int j=0; j<h_patch; j++)
            temp_arr[i][j]=A[i+1][j+1];

        // double rotation of array temp_arr
        for (int i=0; i<h_patch; i++)
          for (int j=0; j<h_patch/2; j++){
              int row_dist=abs(h_patch/2-i);
              int col_dist=abs(h_patch/2-j);
              float temp_el=temp_arr[abs(i-2*(row_dist))][abs(j-2*(col_dist))];
              // swap the elements symmetric to the center
              temp_arr[abs(i-2*(row_dist))][abs(j-2*(col_dist))]=temp_arr[i][j];
              temp_arr[i][j]=temp_el;
          }

        for (int i=0; i<h_patch; i++)
          for (int j=0; j<h_patch; j++)
            Out[i][j]=temp_arr[i][j];

        break;

      case (UR):
        for (int i=0; i<h_patch; i++)
          for (int j=0; j<h_patch; j++)
            temp_arr[i][j]=A[i+1][n-h_patch+j-1];

        // double rotation of array temp_arr
        for (int i=0; i<h_patch; i++)
          for (int j=0; j<h_patch/2; j++){
            int row_dist=abs(h_patch/2-i);
            int col_dist=abs(h_patch/2-j);
            float temp_el=temp_arr[abs(i-2*(row_dist))][abs(j-2*(col_dist))];
            // swap the elements symmetric to the center
            temp_arr[abs(i-2*(row_dist))][abs(j-2*(col_dist))]=temp_arr[i][j];
            temp_arr[i][j]=temp_el;
          }

        // in this case the inverse of temp array is necessary
        for (int i=0; i<h_patch; i++)
          for (int j=0; j<h_patch; j++)
            Out[i][size-j-1]=temp_arr[h_patch-i-1][h_patch-j-1];

        break;

      case (DL):
        for (int i=0; i<h_patch; i++)
          for (int j=0; j<h_patch; j++)
            temp_arr[i][j]=A[n-h_patch+i-1][j+1];

        // double rotation of array temp_arr
        for (int i=0; i<h_patch; i++)
          for (int j=0; j<h_patch/2; j++){
            int row_dist=abs(h_patch/2-i);
            int col_dist=abs(h_patch/2-j);
            float temp_el=temp_arr[abs(i-2*(row_dist))][abs(j-2*(col_dist))];
            // swap the elements symmetric to the center
            temp_arr[abs(i-2*(row_dist))][abs(j-2*(col_dist))]=temp_arr[i][j];
            temp_arr[i][j]=temp_el;
          }
          // in this case the inverse of temp array is necessary
        for (int i=0; i<h_patch; i++)
          for (int j=0; j<h_patch; j++)
            Out[size-i-1][j]=temp_arr[h_patch-1-i][j];

        break;

      case (DR):
        for (int i=0; i<h_patch; i++)
          for (int j=0; j<h_patch; j++)
            temp_arr[i][j]=A[n-h_patch+i-1][n-h_patch+j-1];

            // double rotation of array temp_arr
        for (int i=0; i<h_patch; i++)
          for (int j=0; j<h_patch/2; j++){
            int row_dist=abs(h_patch/2-i);
            int col_dist=abs(h_patch/2-j);
            float temp_el=temp_arr[abs(i-2*(row_dist))][abs(j-2*(col_dist))];
            // swap the elements symmetric to the center
            temp_arr[abs(i-2*(row_dist))][abs(j-2*(col_dist))]=temp_arr[i][j];
            temp_arr[i][j]=temp_el;
          }

        for (int i=0; i<h_patch; i++)
          for (int j=0; j<h_patch; j++)
            Out[size-i-1][size-j-1]=temp_arr[h_patch-1-i][h_patch-1-j];

        break;

      default:
        printf("Oh...switch statement gone wrong");
        break;
      }
    }

    for (int i=0; i<h_patch; i++)
      free(temp_arr[i]);
    free(temp_arr);
    */
    return Out;
  }


void filter(float** A,int n,int patchsize,float patchsigma,float filtsigma){

  int h_patch=(patchsize-1)/2;
  int size=2*h_patch+n;

  // mirroring array A
  float** B=mirroring(A,patchsize,n);
  float** patches=(float**)malloc(n*n*sizeof(float*));
  for (int i=0; i<n*n; i++)
    patches[i]=(float*)malloc(patchsize*patchsize*sizeof(float));

  for (int i=0; i<n; i++)
    for (int j=0; j<n; j++)
      patches[i*n+j]=patch(B,i+h_patch,j+h_patch,patchsize);

  for(int i=0; i<size; i++)
    free(B[i]);
  free(B);

  // calculate distance for each patch
  float** d_array=(float**)malloc(n*n*sizeof(float*));
  for (int i=0; i<n*n; i++)
    d_array[i]=(float*)malloc(n*n*sizeof(float));
  for (int k=0; k<(n*n); k++)
    for (int i=0; i<(n*n); i++){
        if (i==k){
          d_array[k][i]=1;
        }
        else{
          float temp=dist_calc_g(patches[k],patches[i],patchsize,patchsigma);
          d_array[k][i]=exp((-temp)/filtsigma);
        }
  }

  // generating image
  float* temp_out=(float*)malloc(n*n*sizeof(float));
  for(int i=0; i<n; i++)
    for (int j=0; j<n; j++)
      temp_out[i*n+j]=A[i][j];

  float* temp=(float*)calloc(n*n,sizeof(float));
  float sum=0;
  for (int i=0; i<n*n; i++){
    for (int j=0; j<n*n; j++){
        temp[i]+=d_array[i][j]*temp_out[j];
        sum+=d_array[i][j];
      }
    temp[i]/=sum;
    sum=0;
  }

  free(temp_out);
  for (int i=0; i<n*n; i++)
    free(d_array[i]);
  free(d_array);
  for (int i=0; i<n; i++)
    for (int j=0; j<n; j++)
      A[j][i]=temp[j*n+i];

  free(temp);
}







/*

    INPUT SHOULD BE IN : ./script_name ./images/image.txt dim1xdim2
    Image should be converted to .txt file using the matlab code image_convert_2_txt

*/



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



  printf("Done Normilizing!\n");
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

  char* n_img_name=(char*)malloc(strlen(img_name+strlen("_noised.txt"))*sizeof(char));
  if (n_img_name==NULL){
    printf("\nNo memory can be allocated,exiting...\n");
    exit(-1);
  }
  memcpy(n_img_name,img_name,strlen(img_name));
  strcat(n_img_name,"_noised.txt");

  char* img_filt=(char*)malloc(strlen(img_name+strlen("_filtered.txt"))*sizeof(char));
  if (img_filt==NULL){
    printf("\nNo memory can be allocated,exiting...\n");
    exit(-1);
  }
  memcpy(img_filt,img_name,strlen(img_name));
  strcat(img_filt,"_filtered.txt");

  char* img_res=(char*)malloc(strlen(img_name+strlen("_residual.txt"))*sizeof(char));
  if (img_res==NULL){
    printf("\nNo memory can be allocated,exiting...\n");
    exit(-1);
  }
  memcpy(img_res,img_name,strlen(img_name));
  strcat(img_res,"_residual.txt");

*/
  free(img_name);


  //*********************************************

  //                  PARAMETERS

  int patchsize=5;
  float noise_sigma=0.001;
  float patchsigma=5/3;
  float filtsigma=0.02;

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
  //FILE* f_noise=fopen(n_img_name,"w");
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
  //FILE* f_filter=fopen(img_filt,"w");
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
  //FILE* f_ref=fopen(img_res,"w");
  FILE* f_ref=fopen("residual.txt","w");
  for (int i=0; i<rows; i++)
    for (int j=0; j<cols; j++)
      fprintf(f_ref,"%f",fabs(n_img_Arr[i][j]-residual[i][j]));
  fclose(f_ref);

  //*******************************************
  return 0;
}
