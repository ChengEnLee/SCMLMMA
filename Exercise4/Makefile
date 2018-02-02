ALL: GPU CPU
GPU:
	nvcc DenseToCSR.cu -O3 -o gpuRUN
CPU:
	gcc DenseToCSR.c -O3 -o cpuRUN
clean:
	rm gpuRUN cpuRUN
