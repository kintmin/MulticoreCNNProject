#define _CRT_SECURE_NO_WARNINGS
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <io.h>
#include <fcntl.h>
#include "cnn.h"

const int PARALLEL = 60;

#define CHECK_ERROR(err) \
    if(err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

#define MALLOC(p, type, size) \
    if (!(p = (type *)malloc(sizeof(type) * size))) { \
        printf("[%s:%d] malloc error\n", __FILE__, __LINE__);   \
        exit(EXIT_FAILURE); \
    }

#define CHECK_BUILD_ERROR(program) \
	if (err == CL_BUILD_PROGRAM_FAILURE) {	\
		size_t log_size;	\
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);	\
		printf("로그크기: %zu\n", log_size);	\
		char *log;	\
		MALLOC(log, char, log_size);	\
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);	\
		printf("%s\n", log);	\
	}

#define MEM_SWAP(a, b) \
	{ cl_mem *temp = a;	\
	a = b;	\
	b = temp; }

char *GetSourceCode(const char *file_name, size_t *len) {
	int fd;
	char *source_code;
	int cnt = 0;
	size_t length;

	fd = _open(file_name, O_RDONLY);
	if (!fd) {
		printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
		exit(EXIT_FAILURE);
	}

	length = _lseek(fd, 0, SEEK_END);
	MALLOC(source_code, char, length + 1);
	_lseek(fd, 0, SEEK_SET);
	length = _read(fd, source_code, length);
	source_code[length] = '\0';

	_close(fd);
	*len = length;

	return source_code;
}

static void softmax(float *output, int N) {
	int i;
	float max = output[0];
	for (i = 1; i < N; i++) {
		max = (output[i] > max) ? output[i] : max;
	}
	float sum = 0;
	for (i = 0; i < N; i++) {
		sum += exp(output[i] - max);
	}
	for (i = 0; i < N; i++) {
		output[i] = exp(output[i] - max) / sum;
	}
}

static int find_max(float *fc, int N) {
	int i;
	int maxid = 0;
	float maxval = 0;
	for (i = 0; i < N; i++) {
		if (maxval < fc[i]) {
			maxval = fc[i];
			maxid = i;
		}
	}
	return maxid;
}

void get_result(int i, float *output, int *labels, float *confidences) {
	for (int j = 0; j < PARALLEL; ++j) {
		float *result = output + 10 * j;
		softmax(result, 10);
		labels[i + j] = find_max(result, 10);
		confidences[i + j] = result[labels[i + j]];
	}
}

float *alloc_layer(size_t n) {
	return (float *)malloc(n * sizeof(float));
}

cl_platform_id platform;
cl_device_id device;
cl_int err;
cl_context context;
cl_command_queue queue;
cl_program convolution_program;
cl_program pooling_program;
cl_program fc_program;
cl_kernel convolution_kernel, reduction_kernel;
cl_kernel pooling_kernel;
cl_kernel fc_kernel;
cl_mem buf1, buf2,  buf3, buf4, buf_n;
cl_mem *input_buf, *output_buf;

void cnn_init() {
	err = clGetPlatformIDs(1, &platform, NULL);
	CHECK_ERROR(err);

	// device 정보 가져오기 (GPU)
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	CHECK_ERROR(err);

	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	CHECK_ERROR(err);

	queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
	CHECK_ERROR(err);

	size_t source_size;
	const char *source_code = GetSourceCode("convolution_kernel.cl", &source_size);
//	const char *source_code = GetSourceCode("tiled_convolution_kernel.cl", &source_size);
	convolution_program = clCreateProgramWithSource(context, 1, (const char **)&source_code, &source_size, &err);
	CHECK_ERROR(err);

	source_code = GetSourceCode("pooling_kernel.cl", &source_size);
	pooling_program = clCreateProgramWithSource(context, 1, (const char **)&source_code, &source_size, &err);
	CHECK_ERROR(err);

	source_code = GetSourceCode("fc_kernel.cl", &source_size);
	fc_program = clCreateProgramWithSource(context, 1, (const char **)&source_code, &source_size, &err);
	CHECK_ERROR(err);

	err = clBuildProgram(convolution_program, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);
	CHECK_BUILD_ERROR(convolution_program);
	CHECK_ERROR(err);

	err = clBuildProgram(pooling_program, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);
	CHECK_BUILD_ERROR(pooling_program);
	CHECK_ERROR(err);

	err = clBuildProgram(fc_program, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);
	CHECK_BUILD_ERROR(fc_program);
	CHECK_ERROR(err);

	convolution_kernel = clCreateKernel(convolution_program, "convolution", &err);
	CHECK_ERROR(err);

	//reduction_kernel = clCreateKernel(convolution_program, "reduction", &err);
	//CHECK_ERROR(err);

	pooling_kernel = clCreateKernel(pooling_program, "pooling", &err);
	CHECK_ERROR(err);

	fc_kernel = clCreateKernel(fc_program, "fc", &err);
	CHECK_ERROR(err);

	// 4194304 (PARALLEL <= 50)
	// (PARALLEL * 65536)
	buf1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * (PARALLEL * 65536), NULL, &err);    CHECK_ERROR(err);
	buf2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * (PARALLEL * 65536), NULL, &err);    CHECK_ERROR(err);
	buf3 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * 2359296, NULL, &err);    CHECK_ERROR(err);
	buf4 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * 512, NULL, &err);    CHECK_ERROR(err);
	buf_n = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int), NULL, &err);    CHECK_ERROR(err);

	input_buf = &buf1;
	output_buf = &buf2;
}

// input is (P, D1, N, N) and output is (P, D2, N, N)
static void convolution_layer(float *filters, float *biases, int d2, int d1, int n) {
	size_t global_size[] = { PARALLEL * d1, d2 * n * n };
	size_t local_size[] = { d1, 1 };

	clEnqueueWriteBuffer(queue, buf3, CL_TRUE, 0, sizeof(cl_float) * (d2 * d1 * 3 * 3), filters, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, buf4, CL_TRUE, 0, sizeof(cl_float) * d2, biases, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, buf_n, CL_TRUE, 0, sizeof(cl_int), &n, 0, NULL, NULL);

	clSetKernelArg(convolution_kernel, 0, sizeof(cl_mem), input_buf);
	clSetKernelArg(convolution_kernel, 1, sizeof(cl_mem), &buf3);
	clSetKernelArg(convolution_kernel, 2, sizeof(cl_float) * d1, NULL);
	clSetKernelArg(convolution_kernel, 3, sizeof(cl_mem), output_buf);
	clSetKernelArg(convolution_kernel, 4, sizeof(cl_mem), &buf4);
	clSetKernelArg(convolution_kernel, 5, sizeof(cl_mem), &buf_n);
	
	clEnqueueNDRangeKernel(queue, convolution_kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
	clFinish(queue);
	MEM_SWAP(input_buf, output_buf);
}

static void tiled_convolution_layer(float *filters, float *biases, int d2, int d1, int n) {
	size_t global_size[] = { PARALLEL * d2, d1 * n * n };
	size_t local_size[] = { 1, n * n };

	err = clEnqueueWriteBuffer(queue, buf3, CL_TRUE, 0, sizeof(cl_float) * (d2 * d1 * 3 * 3), filters, 0, NULL, NULL);    CHECK_ERROR(err);

	err = clSetKernelArg(convolution_kernel, 0, sizeof(cl_mem), input_buf); CHECK_ERROR(err);
	err = clSetKernelArg(convolution_kernel, 1, sizeof(cl_mem), output_buf); CHECK_ERROR(err);
	err = clSetKernelArg(convolution_kernel, 2, sizeof(cl_mem), &buf3); CHECK_ERROR(err);
	err = clSetKernelArg(convolution_kernel, 3, sizeof(cl_float) * n * n, NULL); CHECK_ERROR(err);
	err = clSetKernelArg(convolution_kernel, 4, sizeof(cl_int), &n); CHECK_ERROR(err);
	err = clSetKernelArg(convolution_kernel, 5, sizeof(cl_int), &d2); CHECK_ERROR(err);

	err = clEnqueueNDRangeKernel(queue, convolution_kernel, 2, NULL, global_size, local_size, 0, NULL, NULL); CHECK_ERROR(err);
	clFinish(queue);
	MEM_SWAP(input_buf, output_buf);

	local_size[1] = d1;
	err = clEnqueueWriteBuffer(queue, buf4, CL_TRUE, 0, sizeof(cl_float) * d2, biases, 0, NULL, NULL);    CHECK_ERROR(err);

	err = clSetKernelArg(reduction_kernel, 0, sizeof(cl_mem), input_buf); CHECK_ERROR(err);
	err = clSetKernelArg(reduction_kernel, 1, sizeof(cl_mem), output_buf); CHECK_ERROR(err);
	err = clSetKernelArg(reduction_kernel, 2, sizeof(cl_mem), &buf4); CHECK_ERROR(err);
	err = clSetKernelArg(reduction_kernel, 3, sizeof(cl_float) * d1, NULL); CHECK_ERROR(err);
	err = clSetKernelArg(reduction_kernel, 4, sizeof(cl_int), &n); CHECK_ERROR(err);
	err = clSetKernelArg(reduction_kernel, 5, sizeof(cl_int), &d2); CHECK_ERROR(err);

	err = clEnqueueNDRangeKernel(queue, reduction_kernel, 2, NULL, global_size, local_size, 0, NULL, NULL); CHECK_ERROR(err);
	clFinish(queue);
	MEM_SWAP(input_buf, output_buf);
}

// input is (P, D, N*2, N*2) and output is (P, D, N, N)
static void pooling_layer(int d, int n) {
	size_t global_size[] = { PARALLEL, d * n * n };

	clSetKernelArg(pooling_kernel, 0, sizeof(cl_mem), input_buf);
	clSetKernelArg(pooling_kernel, 1, sizeof(cl_mem), output_buf);
	clSetKernelArg(pooling_kernel, 2, sizeof(cl_mem), &n);

	clEnqueueNDRangeKernel(queue, pooling_kernel, 2, NULL, global_size, NULL, 0, NULL, NULL);
	clFinish(queue);
	MEM_SWAP(input_buf, output_buf);
}

// input is (P, N) and output is (P, M)
static void fc_layer(float *weights, float *biases, int M, int N) {
	size_t global_size[] = { PARALLEL * M, N };
	size_t local_size[] = { 1, N };

	clEnqueueWriteBuffer(queue, buf3, CL_TRUE, 0, sizeof(cl_float) * M * N, weights, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, buf4, CL_TRUE, 0, sizeof(cl_float) * M, biases, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, buf_n, CL_TRUE, 0, sizeof(cl_int), &PARALLEL, 0, NULL, NULL);

	clSetKernelArg(fc_kernel, 0, sizeof(cl_mem), output_buf);
	clSetKernelArg(fc_kernel, 1, sizeof(cl_mem), input_buf);
	clSetKernelArg(fc_kernel, 2, sizeof(cl_mem), &buf3);
	clSetKernelArg(fc_kernel, 3, sizeof(cl_mem), &buf4);
	clSetKernelArg(fc_kernel, 4, sizeof(cl_float) * N, NULL);
	clSetKernelArg(fc_kernel, 5, sizeof(cl_mem), &buf_n);

	clEnqueueNDRangeKernel(queue, fc_kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
	clFinish(queue);
	MEM_SWAP(input_buf, output_buf);
}

void cnn(float *images, float **network, int *labels, float *confidences, int num_images) {
	float *output = alloc_layer(10 * PARALLEL);

	// run network
	for (int i = 0; i < num_images; i += PARALLEL)
	{
		err = clEnqueueWriteBuffer(queue, *input_buf, CL_TRUE, 0, sizeof(cl_float) * PARALLEL * 3072, images + i * 3072, 0, NULL, NULL);

		convolution_layer(network[0], network[1], 64, 3, 32);
		convolution_layer(network[2], network[3], 64, 64, 32);
		pooling_layer(64, 16);

		convolution_layer(network[4], network[5], 128, 64, 16);
		convolution_layer(network[6], network[7], 128, 128, 16);
		pooling_layer(128, 8);

		convolution_layer(network[8], network[9], 256, 128, 8);
		convolution_layer(network[10], network[11], 256, 256, 8);
		convolution_layer(network[12], network[13], 256, 256, 8);
		pooling_layer(256, 4);

		convolution_layer(network[14], network[15], 512, 256, 4);
		convolution_layer(network[16], network[17], 512, 512, 4);
		convolution_layer(network[18], network[19], 512, 512, 4);
		pooling_layer(512, 2);

		convolution_layer(network[20], network[21], 512, 512, 2);
		convolution_layer(network[22], network[23], 512, 512, 2);
		convolution_layer(network[24], network[25], 512, 512, 2);
		pooling_layer(512, 1);

		fc_layer(network[26], network[27], 512, 512);
		fc_layer(network[28], network[29], 512, 512);
		fc_layer(network[30], network[31], 10, 512);
		
		clEnqueueReadBuffer(queue, *input_buf, CL_TRUE, 0, sizeof(cl_float) * 10 * PARALLEL, output, 0, NULL, NULL);
		get_result(i, output, labels, confidences);
	}
}
