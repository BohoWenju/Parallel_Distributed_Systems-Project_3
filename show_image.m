clear 
close all

% if u want to open an image change filtered.txt
filename="./filtered.txt";
fileId=fopen(filename,'r');
I=fscanf(fileId,"%f");
I=reshape(I,[64 64]);

% funtion to normilize image 
normImg = @(I) (I - min(I(:))) ./ max(I(:) - min(I(:)));

% displaying the image on screen
%I = normImg( I );
  
figure('Name','Image');
rgbImage=I;
imwrite(rgbImage,'image.jpg');
imshow(rgbImage);