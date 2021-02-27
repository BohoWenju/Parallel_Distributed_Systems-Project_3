NVCC = nvcc
CC = gcc
FLAGS= -lm

cpu: Non_local.c
	$(CC) $^ -o Non_local_cpu $(FLAGS)

gpu: Non_local.cu
	$(NVCC) $^ -o Non_local_gpu

gpu_shared: Non_local_shared.cu
	$(NVCC) $^ -o Non_local_shared

all: Non_local.c Non_local.cu  Non_local_shared.cu

	$(CC) Non_local.c -o Non_local_cpu $(FLAGS)

	$(NVCC) Non_local.cu -o Non_local_gpu

	$(NVCC) Non_local_shared.cu -o Non_local_shared

colab: Non_local_colab.cu  Non_local_colab_shared.cu

	$(NVCC) Non_local_colab.cu -o Non_local_colab

	$(NVCC) Non_local_colab_shared.cu -o Non_local_shared_colab

clean:
	rm -f Non_local_cpu  Non_local_gpu Non_local_shared Non_local_colab Non_local_shared_colab

clean_colab:
	rm -f Non_local_colab Non_local_shared_colab
