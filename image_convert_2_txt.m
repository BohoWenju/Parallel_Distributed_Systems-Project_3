  clear 
  close all
  
  % change path img and stringimgvar if u want to edit a new file
  % input image 
  pathImg   = './lady.jpg';
  x=im2double(imread(pathImg));
  x=x(:,:,1);
  x=reshape(x,[128 128]);
  edit .lady.txt
  fid=fopen(".lady.txt",'w');
  fprintf(fid,"%f\t",x);
  fclose(fid);
