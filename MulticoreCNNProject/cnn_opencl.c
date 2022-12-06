#define _CRT_SECURE_NO_WARNINGS
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <io.h>
#include <fcntl.h>
#include "cnn.h"

const int PARALLEL = 2;

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
		printf("�α�ũ��: %zu\n", log_size);	\
		char *log;	\
		MALLOC(log, char, log_size);	\
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);	\
		printf("%s\n", log);	\
	}

char* GetSourceCode(const char* file_name, size_t* len) {
	int fd;
	char* source_code;
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

static void softmax(float* output, int N) {
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

static int find_max(float* fc, int N) {
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

float* alloc_layer(size_t n) {
	return (float*)malloc(n * sizeof(float));
}

cl_platform_id platform;
cl_device_id device;
cl_int err;
cl_context context;
cl_command_queue queue, write_queue, kernel_queue;
cl_program convolution_program;
cl_program pooling_program;
cl_program fc_program;
cl_kernel convolution_kernel, convolution_kernel2;
cl_mem buf_input, buf_output, buf_output2, buf_filter, buf_bias;

cl_kernel pooling_kernel;
cl_kernel fc_kernel;

void cnn_init() {
	err = clGetPlatformIDs(1, &platform, NULL);
	CHECK_ERROR(err);

	// device ���� �������� (GPU)
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	CHECK_ERROR(err);

	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	CHECK_ERROR(err);

	kernel_queue = clCreateCommandQueueWithProperties(context, device, 0, &err);  CHECK_ERROR(err);
	write_queue = clCreateCommandQueueWithProperties(context, device, 0, &err);  CHECK_ERROR(err);
	queue = clCreateCommandQueueWithProperties(context, device, 0, &err);  CHECK_ERROR(err);

	size_t source_size;
	const char* source_code = GetSourceCode("convolution_kernel.cl", &source_size);
	convolution_program = clCreateProgramWithSource(context, 1, (const char**)&source_code, &source_size, &err);
	CHECK_ERROR(err);

	source_code = GetSourceCode("pooling_kernel.cl", &source_size);
	pooling_program = clCreateProgramWithSource(context, 1, (const char**)&source_code, &source_size, &err);
	CHECK_ERROR(err);

	source_code = GetSourceCode("fc_kernel.cl", &source_size);
	fc_program = clCreateProgramWithSource(context, 1, (const char**)&source_code, &source_size, &err);
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

	convolution_kernel = clCreateKernel(convolution_program, "convolution", &err);	CHECK_ERROR(err);
	convolution_kernel2 = clCreateKernel(convolution_program, "convolution", &err);	CHECK_ERROR(err);

	pooling_kernel = clCreateKernel(pooling_program, "pooling", &err);
	CHECK_ERROR(err);

	fc_kernel = clCreateKernel(fc_program, "fc", &err);
	CHECK_ERROR(err);
}

// input is (P, D1, N, N) and output is (P, D2, N, N)
static void convolution_layer(float* inputs, float* outputs, float* filters, float* biases, int d2, int d1, int n) {
	int num_buffering = 2;

	buf_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * (PARALLEL * d1 * n * n), inputs, &err);	CHECK_ERROR(err);
	buf_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * (PARALLEL / num_buffering  * d2 * n * n), NULL, &err);	CHECK_ERROR(err);
	buf_output2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * (PARALLEL / num_buffering * d2 * n * n), NULL, &err);	CHECK_ERROR(err);
	buf_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * (d2 * d1 * 3 * 3), filters, &err);	CHECK_ERROR(err);
	buf_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * d2, biases, &err);	CHECK_ERROR(err)

	err = clSetKernelArg(convolution_kernel, 0, sizeof(cl_mem), &buf_input);	CHECK_ERROR(err);
	err = clSetKernelArg(convolution_kernel, 1, sizeof(cl_mem), &buf_filter);	CHECK_ERROR(err);
	err = clSetKernelArg(convolution_kernel, 2, sizeof(cl_float) * d1, NULL);	CHECK_ERROR(err);
	err = clSetKernelArg(convolution_kernel, 3, sizeof(cl_mem), &buf_output);	CHECK_ERROR(err);
	err = clSetKernelArg(convolution_kernel, 4, sizeof(cl_mem), &buf_bias);	CHECK_ERROR(err);
	err = clSetKernelArg(convolution_kernel, 5, sizeof(cl_int), &n);	CHECK_ERROR(err);

	err = clSetKernelArg(convolution_kernel2, 0, sizeof(cl_mem), &buf_input);	CHECK_ERROR(err);
	err = clSetKernelArg(convolution_kernel2, 1, sizeof(cl_mem), &buf_filter);	CHECK_ERROR(err);
	err = clSetKernelArg(convolution_kernel2, 2, sizeof(cl_float) * d1, NULL);	CHECK_ERROR(err);
	err = clSetKernelArg(convolution_kernel2, 3, sizeof(cl_mem), &buf_output2);	CHECK_ERROR(err);
	err = clSetKernelArg(convolution_kernel2, 4, sizeof(cl_mem), &buf_bias);	CHECK_ERROR(err);
	err = clSetKernelArg(convolution_kernel2, 5, sizeof(cl_int), &n);	CHECK_ERROR(err);


	size_t global_size[] = { d1 * PARALLEL / num_buffering, d2 * n * n };
	size_t local_size[] = { d1, 1 };
	cl_event kernel_event[4] = { NULL, NULL, NULL, NULL };
	for (int i = 0; i < num_buffering / 2; i++) {
		int j = i * PARALLEL / num_buffering;
		int k = (i + 1) * PARALLEL / num_buffering;

		err = clSetKernelArg(convolution_kernel, 6, sizeof(cl_int), &j);	CHECK_ERROR(err);
		err = clSetKernelArg(convolution_kernel2, 6, sizeof(cl_int), &k);	CHECK_ERROR(err);
		if (kernel_event[2] != NULL)
			err = clEnqueueNDRangeKernel(kernel_queue, convolution_kernel, 2, NULL, global_size, local_size, 1, &kernel_event[2], &kernel_event[0]);
		else 
			err = clEnqueueNDRangeKernel(kernel_queue, convolution_kernel, 2, NULL, global_size, local_size, 0, NULL, &kernel_event[0]);
		CHECK_ERROR(err);
		err = clEnqueueReadBuffer(write_queue, buf_output, CL_FALSE, 0, sizeof(cl_float) * (PARALLEL / num_buffering * d2 * n * n), outputs, 1, &kernel_event[0], &kernel_event[1]);	CHECK_ERROR(err);

		err = clEnqueueNDRangeKernel(kernel_queue, convolution_kernel2, 2, NULL, global_size, local_size, 1, &kernel_event[0], &kernel_event[2]);	CHECK_ERROR(err);
		float* output = outputs + PARALLEL / num_buffering * d2 * n * n;
		err = clEnqueueReadBuffer(write_queue, buf_output2, CL_FALSE, 0, 
			sizeof(cl_float) * (PARALLEL / num_buffering * d2 * n * n), output, 1, &kernel_event[2], &kernel_event[3]);	CHECK_ERROR(err);
	}

	clFinish(write_queue);

	clReleaseMemObject(buf_input);
	clReleaseMemObject(buf_output);
	clReleaseMemObject(buf_output2);
	clReleaseMemObject(buf_filter);
	clReleaseMemObject(buf_bias);
}

// input is (P, D, N*2, N*2) and output is (P, D, N, N)
static void pooling_layer(float* inputs, float* outputs, int d, int n) {
	size_t global_size[] = { PARALLEL, d * n * n };

	cl_mem buf_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * (PARALLEL * d * n * n * 4), inputs, &err);
	CHECK_ERROR(err);

	cl_mem buf_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * (PARALLEL * d * n * n), NULL, &err);
	CHECK_ERROR(err);

	err = clSetKernelArg(pooling_kernel, 0, sizeof(cl_mem), &buf_input);
	CHECK_ERROR(err);

	err = clSetKernelArg(pooling_kernel, 1, sizeof(cl_mem), &buf_output);
	CHECK_ERROR(err);

	err = clSetKernelArg(pooling_kernel, 2, sizeof(cl_int), &n);
	CHECK_ERROR(err);

	err = clEnqueueNDRangeKernel(queue, pooling_kernel, 2, NULL, global_size, NULL, 0, NULL, NULL);
	CHECK_ERROR(err);

	err = clEnqueueReadBuffer(queue, buf_output, CL_TRUE, 0, sizeof(cl_float) * (PARALLEL * d * n * n), outputs, 0, NULL, NULL);
	CHECK_ERROR(err);

	clFinish(queue);

	clReleaseMemObject(buf_input);
	clReleaseMemObject(buf_output);
}

// input is (P, N) and output is (P, M)
static void fc_layer(float* input_neuron, float* output_neuron, float* weights, float* biases, int M, int N) {
	size_t global_size[] = { PARALLEL * M, N };
	size_t local_size[] = { 1, N };

	cl_mem buf_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * (PARALLEL * M), NULL, &err);
	CHECK_ERROR(err);

	cl_mem buf_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * (PARALLEL * N), input_neuron, &err);
	CHECK_ERROR(err);

	cl_mem buf_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * M * N, weights, &err);
	CHECK_ERROR(err);

	cl_mem buf_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * M, biases, &err);
	CHECK_ERROR(err);

	err = clSetKernelArg(fc_kernel, 0, sizeof(cl_mem), &buf_output);
	CHECK_ERROR(err);

	err = clSetKernelArg(fc_kernel, 1, sizeof(cl_mem), &buf_input);
	CHECK_ERROR(err);

	err = clSetKernelArg(fc_kernel, 2, sizeof(cl_mem), &buf_weight);
	CHECK_ERROR(err);

	err = clSetKernelArg(fc_kernel, 3, sizeof(cl_mem), &buf_bias);
	CHECK_ERROR(err);

	err = clSetKernelArg(fc_kernel, 4, sizeof(cl_float) * N, NULL);
	CHECK_ERROR(err);

	err = clSetKernelArg(fc_kernel, 5, sizeof(cl_int), &PARALLEL);
	CHECK_ERROR(err);

	err = clEnqueueNDRangeKernel(queue, fc_kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
	CHECK_ERROR(err);

	err = clEnqueueReadBuffer(queue, buf_output, CL_TRUE, 0, sizeof(cl_float) * M * PARALLEL, output_neuron, 0, NULL, NULL);
	CHECK_ERROR(err);

	clFinish(queue);

	clReleaseMemObject(buf_input);
	clReleaseMemObject(buf_output);
	clReleaseMemObject(buf_weight);
	clReleaseMemObject(buf_bias);
}

void cnn(float* images, float** network, int* labels, float* confidences, int num_images) {
	// slice the network into weights and biases
	float* w1_1, * b1_1, * w1_2, * b1_2;
	float* w2_1, * b2_1, * w2_2, * b2_2;
	float* w3_1, * b3_1, * w3_2, * b3_2, * w3_3, * b3_3;
	float* w4_1, * b4_1, * w4_2, * b4_2, * w4_3, * b4_3;
	float* w5_1, * b5_1, * w5_2, * b5_2, * w5_3, * b5_3;
	float* w1, * b1, * w2, * b2, * w3, * b3;
	w1_1 = network[0]; b1_1 = network[1];
	w1_2 = network[2]; b1_2 = network[3];
	w2_1 = network[4]; b2_1 = network[5];
	w2_2 = network[6]; b2_2 = network[7];
	w3_1 = network[8]; b3_1 = network[9];
	w3_2 = network[10]; b3_2 = network[11];
	w3_3 = network[12]; b3_3 = network[13];
	w4_1 = network[14]; b4_1 = network[15];
	w4_2 = network[16]; b4_2 = network[17];
	w4_3 = network[18]; b4_3 = network[19];
	w5_1 = network[20]; b5_1 = network[21];
	w5_2 = network[22]; b5_2 = network[23];
	w5_3 = network[24]; b5_3 = network[25];
	w1 = network[26]; b1 = network[27];
	w2 = network[28]; b2 = network[29];
	w3 = network[30]; b3 = network[31];

	// allocate memory for output of each layer
	float* c1_1, * c1_2, * p1;
	float* c2_1, * c2_2, * p2;
	float* c3_1, * c3_2, * c3_3, * p3;
	float* c4_1, * c4_2, * c4_3, * p4;
	float* c5_1, * c5_2, * c5_3, * p5;
	float* fc1, * fc2, * fc3;
	c1_1 = alloc_layer(64 * 32 * 32 * PARALLEL);
	c1_2 = alloc_layer(64 * 32 * 32 * PARALLEL);
	p1 = alloc_layer(64 * 16 * 16 * PARALLEL);
	c2_1 = alloc_layer(128 * 16 * 16 * PARALLEL);
	c2_2 = alloc_layer(128 * 16 * 16 * PARALLEL);
	p2 = alloc_layer(128 * 8 * 8 * PARALLEL);
	c3_1 = alloc_layer(256 * 8 * 8 * PARALLEL);
	c3_2 = alloc_layer(256 * 8 * 8 * PARALLEL);
	c3_3 = alloc_layer(256 * 8 * 8 * PARALLEL);
	p3 = alloc_layer(256 * 4 * 4 * PARALLEL);
	c4_1 = alloc_layer(512 * 4 * 4 * PARALLEL);
	c4_2 = alloc_layer(512 * 4 * 4 * PARALLEL);
	c4_3 = alloc_layer(512 * 4 * 4 * PARALLEL);
	p4 = alloc_layer(512 * 2 * 2 * PARALLEL);
	c5_1 = alloc_layer(512 * 2 * 2 * PARALLEL);
	c5_2 = alloc_layer(512 * 2 * 2 * PARALLEL);
	c5_3 = alloc_layer(512 * 2 * 2 * PARALLEL);
	p5 = alloc_layer(512 * 1 * 1 * PARALLEL);
	fc1 = alloc_layer(512 * PARALLEL);
	fc2 = alloc_layer(512 * PARALLEL);
	fc3 = alloc_layer(10 * PARALLEL);

	// run network
	for (int i = 0; i < num_images; i += PARALLEL)
	{
		float* image = images + i * 3 * 32 * 32;

		convolution_layer(image, c1_1, w1_1, b1_1, 64, 3, 32);
		convolution_layer(c1_1, c1_2, w1_2, b1_2, 64, 64, 32);
		pooling_layer(c1_2, p1, 64, 16);

		convolution_layer(p1, c2_1, w2_1, b2_1, 128, 64, 16);
		convolution_layer(c2_1, c2_2, w2_2, b2_2, 128, 128, 16);
		pooling_layer(c2_2, p2, 128, 8);

		convolution_layer(p2, c3_1, w3_1, b3_1, 256, 128, 8);
		convolution_layer(c3_1, c3_2, w3_2, b3_2, 256, 256, 8);
		convolution_layer(c3_2, c3_3, w3_3, b3_3, 256, 256, 8);
		pooling_layer(c3_3, p3, 256, 4);

		convolution_layer(p3, c4_1, w4_1, b4_1, 512, 256, 4);
		convolution_layer(c4_1, c4_2, w4_2, b4_2, 512, 512, 4);
		convolution_layer(c4_2, c4_3, w4_3, b4_3, 512, 512, 4);
		pooling_layer(c4_3, p4, 512, 2);

		convolution_layer(p4, c5_1, w5_1, b5_1, 512, 512, 2);
		convolution_layer(c5_1, c5_2, w5_2, b5_2, 512, 512, 2);
		convolution_layer(c5_2, c5_3, w5_3, b5_3, 512, 512, 2);
		pooling_layer(c5_3, p5, 512, 1);

		fc_layer(p5, fc1, w1, b1, 512, 512);
		fc_layer(fc1, fc2, w2, b2, 512, 512);
		fc_layer(fc2, fc3, w3, b3, 10, 512);

		float* result;
		for (int j = 0; j < PARALLEL; ++j) {
			result = fc3 + 10 * j;
			softmax(result, 10);
			labels[i + j] = find_max(result, 10);
			confidences[i + j] = result[labels[i + j]];
		}
	}

	free(c1_1); free(c1_2); free(p1);
	free(c2_1); free(c2_2); free(p2);
	free(c3_1); free(c3_2); free(c3_3); free(p3);
	free(c4_1); free(c4_2); free(c4_3); free(p4);
	free(c5_1); free(c5_2); free(c5_3); free(p5);
	free(fc1); free(fc2); free(fc3);
}