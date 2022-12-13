
#define _CRT_SECURE_NO_WARNINGS
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <io.h>
#include <fcntl.h>
#include "cnn.h"
#include <time.h>

const int PARALLEL = 1000;
const int num_buffering = 20;
const int batch_num = 50;

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
if (err == CL_BUILD_PROGRAM_FAILURE) { \
size_t log_size; \
clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size); \
printf("�α�ũ��: %zu\n", log_size); \
char *log; \
MALLOC(log, char, log_size); \
clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL); \
printf("%s\n", log); \
}


const int INPUT_DIM[] = {
	3, 64,
	64,

	64,128,
	128,

	128, 256, 256,
	256,

	256, 512, 512,
	512,

	512, 512, 512,
	512,

	512,
	512,
	512
};

const int OUTPUT_DIM[] = {
	64, 64,
	64,

	128, 128,
	128,

	256, 256, 256,
	256,

	512, 512, 512,
	512,

	512, 512, 512,
	512,

	512,
	512,
	10
};

const int NBYN[] = {
	32, 32,
	16,

	16, 16,
	8,

	8, 8, 8,
	4,

	4, 4, 4,
	2,

	2, 2, 2,
	1,

	1,
	1,
	1
};

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
cl_program convolution_program, convolution_program_2;
cl_program pooling_program;
cl_program fc_program;
cl_kernel convolution_kernel, convolution_kernel2;
cl_kernel convolution_kernel_2, convolution_kernel_22;
cl_kernel pooling_kernel, pooling_kernel2;
cl_kernel fc_kernel, fc_kernel2;
cl_mem buf1, buf1_1, buf2, buf2_1, buf3, buf4;

void cnn_init() {
	err = clGetPlatformIDs(1, &platform, NULL);
	CHECK_ERROR(err);
	// device ���� �������� (GPU)
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	CHECK_ERROR(err);

	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	CHECK_ERROR(err);

	queue = clCreateCommandQueueWithProperties(context, device, 0, &err);    CHECK_ERROR(err);
	write_queue = clCreateCommandQueueWithProperties(context, device, 0, &err);    CHECK_ERROR(err);
	kernel_queue = clCreateCommandQueueWithProperties(context, device, 0, &err);    CHECK_ERROR(err);

	size_t source_size;
	const char* source_code = GetSourceCode("convolution_kernel.cl", &source_size);
	convolution_program = clCreateProgramWithSource(context, 1, (const char**)&source_code, &source_size, &err);
	CHECK_ERROR(err);

	source_code = GetSourceCode("convolution_kernel_odd.cl", &source_size);
	convolution_program_2 = clCreateProgramWithSource(context, 1, (const char**)&source_code, &source_size, &err);
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

	err = clBuildProgram(convolution_program_2, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);
	CHECK_BUILD_ERROR(convolution_program_2);
	CHECK_ERROR(err);

	err = clBuildProgram(pooling_program, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);
	CHECK_BUILD_ERROR(pooling_program);
	CHECK_ERROR(err);

	err = clBuildProgram(fc_program, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);
	CHECK_BUILD_ERROR(fc_program);
	CHECK_ERROR(err);

	convolution_kernel = clCreateKernel(convolution_program, "convolution", &err);    CHECK_ERROR(err);
	convolution_kernel2 = clCreateKernel(convolution_program, "convolution", &err);    CHECK_ERROR(err);

	convolution_kernel_2 = clCreateKernel(convolution_program_2, "convolution", &err);    CHECK_ERROR(err);
	convolution_kernel_22 = clCreateKernel(convolution_program_2, "convolution", &err);    CHECK_ERROR(err);

	pooling_kernel = clCreateKernel(pooling_program, "pooling", &err);    CHECK_ERROR(err);
	pooling_kernel2 = clCreateKernel(pooling_program, "pooling", &err);    CHECK_ERROR(err);

	fc_kernel = clCreateKernel(fc_program, "fc", &err);    CHECK_ERROR(err);
	fc_kernel2 = clCreateKernel(fc_program, "fc", &err);    CHECK_ERROR(err);

	buf1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * (PARALLEL * 65536), NULL, &err);    CHECK_ERROR(err);
	buf1_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * (PARALLEL * 65536), NULL, &err);    CHECK_ERROR(err);

	buf2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * (PARALLEL * 65536), NULL, &err);    CHECK_ERROR(err);
	buf2_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * (PARALLEL * 65536), NULL, &err);    CHECK_ERROR(err);

	buf3 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * (2359296), NULL, &err);    CHECK_ERROR(err);

	buf4 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * (512), NULL, &err);    CHECK_ERROR(err);
}

// input is (P, D1, N, N) and output is (P, D2, N, N)
static void convolution_layer(float* inputs, float* outputs, float* filters, float* biases, int d1, int d2, int n) {
	if (d1 < 0) {
		err = clEnqueueWriteBuffer(kernel_queue, buf3, CL_TRUE, 0, sizeof(cl_float) * (d2 * d1 * 3 * 3), filters, 0, NULL, NULL);    CHECK_ERROR(err);
		err = clEnqueueWriteBuffer(kernel_queue, buf4, CL_TRUE, 0, sizeof(cl_float) * d2, biases, 0, NULL, NULL);    CHECK_ERROR(err);

		err = clSetKernelArg(convolution_kernel, 1, sizeof(cl_float) * d1 * 9, NULL);    CHECK_ERROR(err);
		err = clSetKernelArg(convolution_kernel, 2, sizeof(cl_mem), &buf2);    CHECK_ERROR(err);
		err = clSetKernelArg(convolution_kernel, 3, sizeof(cl_mem), &buf4);    CHECK_ERROR(err);
		err = clSetKernelArg(convolution_kernel, 4, sizeof(cl_mem), &buf3);    CHECK_ERROR(err);
		err = clSetKernelArg(convolution_kernel, 5, sizeof(cl_int), &n);    CHECK_ERROR(err);
		err = clSetKernelArg(convolution_kernel, 6, sizeof(cl_int), &d2);    CHECK_ERROR(err);

		err = clSetKernelArg(convolution_kernel2, 1, sizeof(cl_float) * d1 * 9, NULL);    CHECK_ERROR(err);
		err = clSetKernelArg(convolution_kernel2, 2, sizeof(cl_mem), &buf2_1);    CHECK_ERROR(err);
		err = clSetKernelArg(convolution_kernel2, 3, sizeof(cl_mem), &buf4);    CHECK_ERROR(err);
		err = clSetKernelArg(convolution_kernel2, 4, sizeof(cl_mem), &buf3);    CHECK_ERROR(err);
		err = clSetKernelArg(convolution_kernel2, 5, sizeof(cl_int), &n);    CHECK_ERROR(err);
		err = clSetKernelArg(convolution_kernel2, 6, sizeof(cl_int), &d2);    CHECK_ERROR(err);

		size_t global_size[] = { d1 * 9 , d2 * n * n * batch_num };
		size_t local_size[] = { d1 * 9, 1 };
		cl_event kernel_event[4] = { NULL };
		for (int i = 0; i < num_buffering; i += 2) {
			float* input1 = inputs + i * batch_num * d1 * n * n;
			float* output1 = outputs + i * batch_num * d2 * n * n;

			err = clEnqueueWriteBuffer(kernel_queue, buf1, CL_TRUE, 0, sizeof(cl_float) * (batch_num * d1 * n * n), input1, 0, NULL, NULL);    CHECK_ERROR(err);
			err = clSetKernelArg(convolution_kernel, 0, sizeof(cl_mem), &buf1);    CHECK_ERROR(err);
			if (kernel_event[2] != NULL)
				err = clEnqueueNDRangeKernel(kernel_queue, convolution_kernel, 2, NULL, global_size, local_size, 1, &kernel_event[2], &kernel_event[0]);
			else
				err = clEnqueueNDRangeKernel(kernel_queue, convolution_kernel, 2, NULL, global_size, local_size, 0, NULL, &kernel_event[0]);        CHECK_ERROR(err);
			err = clEnqueueReadBuffer(write_queue, buf2, CL_FALSE, 0, sizeof(cl_float) * (batch_num * d2 * n * n), output1, 1, &kernel_event[0], &kernel_event[1]);	CHECK_ERROR(err);

			////////kernel2///////
			int k = i + 1;
			float* input2 = inputs + k * batch_num * d1 * n * n;
			float* output2 = outputs + k * batch_num * d2 * n * n;

			err = clEnqueueWriteBuffer(kernel_queue, buf1_1, CL_TRUE, 0, sizeof(cl_float) * (batch_num * d1 * n * n), input2, 0, NULL, NULL);    CHECK_ERROR(err);
			err = clSetKernelArg(convolution_kernel2, 0, sizeof(cl_mem), &buf1_1);    CHECK_ERROR(err);
			err = clEnqueueNDRangeKernel(kernel_queue, convolution_kernel2, 2, NULL, global_size, local_size, 1, &kernel_event[0], &kernel_event[2]);	CHECK_ERROR(err);
			err = clEnqueueReadBuffer(write_queue, buf2_1, CL_FALSE, 0, sizeof(cl_float) * (batch_num * d2 * n * n), output2, 1, &kernel_event[2], &kernel_event[3]);	CHECK_ERROR(err);
		}
	}
	else {

		err = clEnqueueWriteBuffer(kernel_queue, buf3, CL_TRUE, 0, sizeof(cl_float) * (d2 * d1 * 3 * 3), filters, 0, NULL, NULL);    CHECK_ERROR(err);
		err = clEnqueueWriteBuffer(kernel_queue, buf4, CL_TRUE, 0, sizeof(cl_float) * d2, biases, 0, NULL, NULL);    CHECK_ERROR(err);

		err = clSetKernelArg(convolution_kernel_2, 1, sizeof(cl_mem), &buf3);    CHECK_ERROR(err);
		err = clSetKernelArg(convolution_kernel_2, 2, sizeof(cl_float) * d1, NULL);    CHECK_ERROR(err);
		err = clSetKernelArg(convolution_kernel_2, 3, sizeof(cl_mem), &buf2);    CHECK_ERROR(err);
		err = clSetKernelArg(convolution_kernel_2, 4, sizeof(cl_mem), &buf4);    CHECK_ERROR(err);
		err = clSetKernelArg(convolution_kernel_2, 5, sizeof(cl_int), &n);    CHECK_ERROR(err);

		err = clSetKernelArg(convolution_kernel_22, 1, sizeof(cl_mem), &buf3);    CHECK_ERROR(err);
		err = clSetKernelArg(convolution_kernel_22, 2, sizeof(cl_float) * d1, NULL);    CHECK_ERROR(err);
		err = clSetKernelArg(convolution_kernel_22, 3, sizeof(cl_mem), &buf2_1);    CHECK_ERROR(err);
		err = clSetKernelArg(convolution_kernel_22, 4, sizeof(cl_mem), &buf4);    CHECK_ERROR(err);
		err = clSetKernelArg(convolution_kernel_22, 5, sizeof(cl_int), &n);    CHECK_ERROR(err);

		size_t global_size[] = { d1 * batch_num, d2 * n * n };
		size_t local_size[] = { d1, 1 };
		cl_event kernel_event[4] = { NULL, NULL, NULL, NULL };
		for (int i = 0; i < num_buffering; i += 2) {
			int k = i + 1;

			float* input1 = inputs + i * batch_num * d1 * n * n;
			float* output1 = outputs + i * batch_num * d2 * n * n;
			err = clEnqueueWriteBuffer(kernel_queue, buf1, CL_TRUE, 0, sizeof(cl_float) * (batch_num * d1 * n * n), input1, 0, NULL, NULL);    CHECK_ERROR(err);
			err = clSetKernelArg(convolution_kernel_2, 0, sizeof(cl_mem), &buf1);    CHECK_ERROR(err);
			if (kernel_event[2] != NULL)
				err = clEnqueueNDRangeKernel(kernel_queue, convolution_kernel_2, 2, NULL, global_size, local_size, 1, &kernel_event[2], &kernel_event[0]);
			else
				err = clEnqueueNDRangeKernel(kernel_queue, convolution_kernel_2, 2, NULL, global_size, local_size, 0, NULL, &kernel_event[0]);        CHECK_ERROR(err);
			time_t start, end;
			start = clock();

			err = clEnqueueReadBuffer(write_queue, buf2, CL_FALSE, 0, sizeof(cl_float) * (batch_num * d2 * n * n), output1, 1, &kernel_event[0], &kernel_event[1]);	CHECK_ERROR(err);
			end = clock();
			printf("Elapsed time: %f sec\n", (double)(end - start) / CLK_TCK);
			////////kernel2///////
			float* input2 = inputs + k * batch_num * d1 * n * n;
			float* output2 = outputs + k * batch_num * d2 * n * n;
			err = clEnqueueWriteBuffer(kernel_queue, buf1_1, CL_TRUE, 0, sizeof(cl_float) * (batch_num * d1 * n * n), input2, 0, NULL, NULL);    CHECK_ERROR(err);
			err = clSetKernelArg(convolution_kernel_22, 0, sizeof(cl_mem), &buf1_1);    CHECK_ERROR(err);
			err = clEnqueueNDRangeKernel(kernel_queue, convolution_kernel_22, 2, NULL, global_size, local_size, 1, &kernel_event[0], &kernel_event[2]);	CHECK_ERROR(err);
			err = clEnqueueReadBuffer(write_queue, buf2_1, CL_FALSE, 0,
				sizeof(cl_float) * (batch_num * d2 * n * n), output2, 1, &kernel_event[2], &kernel_event[3]);	CHECK_ERROR(err);
		}
	}

	clFinish(write_queue);
}

// input is (P, D, N*2, N*2) and output is (P, D, N, N)
static void pooling_layer(float* inputs, float* outputs, int d, int n) {
	size_t global_size[] = { batch_num, d * n * n };

	err = clSetKernelArg(pooling_kernel, 1, sizeof(cl_mem), &buf2);    CHECK_ERROR(err);
	err = clSetKernelArg(pooling_kernel, 2, sizeof(cl_int), &n);

	err = clSetKernelArg(pooling_kernel2, 1, sizeof(cl_mem), &buf2_1);    CHECK_ERROR(err);
	err = clSetKernelArg(pooling_kernel2, 2, sizeof(cl_int), &n); CHECK_ERROR(err);

	cl_event kernel_event[4] = { NULL, NULL, NULL, NULL };
	for (int i = 0; i < num_buffering; i += 2) {
		int k = i + 1;

		float* input1 = inputs + i * batch_num * d * n * n * 4;
		float* output1 = outputs + i * batch_num * d * n * n;
		err = clEnqueueWriteBuffer(kernel_queue, buf1, CL_TRUE, 0, sizeof(cl_float) * (batch_num * d * n * n * 4), input1, 0, NULL, NULL);    CHECK_ERROR(err);
		err = clSetKernelArg(pooling_kernel, 0, sizeof(cl_mem), &buf1);    CHECK_ERROR(err);
		if (kernel_event[2] != NULL)
			err = clEnqueueNDRangeKernel(kernel_queue, pooling_kernel, 2, NULL, global_size, NULL, 1, &kernel_event[2], &kernel_event[0]);
		else
			err = clEnqueueNDRangeKernel(kernel_queue, pooling_kernel, 2, NULL, global_size, NULL, 0, NULL, &kernel_event[0]);        CHECK_ERROR(err);
		err = clEnqueueReadBuffer(write_queue, buf2, CL_FALSE, 0, sizeof(cl_float) * (batch_num * d * n * n), output1, 1, &kernel_event[0], &kernel_event[1]);	CHECK_ERROR(err);

		////////kernel2///////
		float* input2 = inputs + k * batch_num * d * n * n * 4;
		float* output2 = outputs + k * batch_num * d * n * n;
		err = clEnqueueWriteBuffer(kernel_queue, buf1_1, CL_TRUE, 0, sizeof(cl_float) * (batch_num * d * n * n * 4), input2, 0, NULL, NULL);    CHECK_ERROR(err);
		err = clSetKernelArg(pooling_kernel2, 0, sizeof(cl_mem), &buf1_1);    CHECK_ERROR(err);
		err = clEnqueueNDRangeKernel(kernel_queue, pooling_kernel2, 2, NULL, global_size, NULL, 1, &kernel_event[0], &kernel_event[2]);	CHECK_ERROR(err);
		err = clEnqueueReadBuffer(write_queue, buf2_1, CL_FALSE, 0,
			sizeof(cl_float) * (batch_num * d * n * n), output2, 1, &kernel_event[2], &kernel_event[3]);	CHECK_ERROR(err);
	}

	clFinish(queue);
}

// input is (P, N) and output is (P, M)
static void fc_layer(float* input_neuron, float* output_neuron, float* weights, float* biases, int N, int M) {
	size_t global_size[] = { batch_num * M, N };
	size_t local_size[] = { 1, N };

	err = clEnqueueWriteBuffer(queue, buf3, CL_TRUE, 0, sizeof(cl_float) * M * N, weights, 0, NULL, NULL);
	CHECK_ERROR(err);

	err = clEnqueueWriteBuffer(queue, buf4, CL_TRUE, 0, sizeof(cl_float) * M, biases, 0, NULL, NULL);
	CHECK_ERROR(err);

	err = clSetKernelArg(fc_kernel, 0, sizeof(cl_mem), &buf2);    CHECK_ERROR(err);
	err = clSetKernelArg(fc_kernel, 2, sizeof(cl_mem), &buf3);    CHECK_ERROR(err);
	err = clSetKernelArg(fc_kernel, 3, sizeof(cl_mem), &buf4);    CHECK_ERROR(err);
	err = clSetKernelArg(fc_kernel, 4, sizeof(cl_float) * N, NULL);    CHECK_ERROR(err);
	err = clSetKernelArg(fc_kernel, 5, sizeof(cl_int), &batch_num);    CHECK_ERROR(err);

	err = clSetKernelArg(fc_kernel2, 0, sizeof(cl_mem), &buf2_1);    CHECK_ERROR(err);
	err = clSetKernelArg(fc_kernel2, 2, sizeof(cl_mem), &buf3);    CHECK_ERROR(err);
	err = clSetKernelArg(fc_kernel2, 3, sizeof(cl_mem), &buf4);    CHECK_ERROR(err);
	err = clSetKernelArg(fc_kernel2, 4, sizeof(cl_float) * N, NULL);    CHECK_ERROR(err);
	err = clSetKernelArg(fc_kernel2, 5, sizeof(cl_int), &batch_num);    CHECK_ERROR(err);

	CHECK_ERROR(err);
	cl_event kernel_event[4] = { NULL, NULL, NULL, NULL };
	for (int i = 0; i < num_buffering; i += 2) {
		int k = i + 1;

		float* input1 = input_neuron + i * batch_num * N;
		float* output1 = output_neuron + i * batch_num * M;
		err = clEnqueueWriteBuffer(kernel_queue, buf1, CL_TRUE, 0, sizeof(cl_float) * (batch_num * N), input1, 0, NULL, NULL);    CHECK_ERROR(err);
		err = clSetKernelArg(fc_kernel, 1, sizeof(cl_mem), &buf1);    CHECK_ERROR(err);
		if (kernel_event[2] != NULL)
			err = clEnqueueNDRangeKernel(kernel_queue, fc_kernel, 2, NULL, global_size, local_size, 1, &kernel_event[2], &kernel_event[0]);
		else
			err = clEnqueueNDRangeKernel(kernel_queue, fc_kernel, 2, NULL, global_size, local_size, 0, NULL, &kernel_event[0]);        CHECK_ERROR(err);
		err = clEnqueueReadBuffer(write_queue, buf2, CL_FALSE, 0, sizeof(cl_float) * (batch_num * M), output1, 1, &kernel_event[0], &kernel_event[1]);	CHECK_ERROR(err);

		////////kernel2///////
		float* input2 = input_neuron + k * batch_num * N;
		float* output2 = output_neuron + k * batch_num * M;
		err = clEnqueueWriteBuffer(kernel_queue, buf1_1, CL_TRUE, 0, sizeof(cl_float) * (batch_num * N), input2, 0, NULL, NULL);    CHECK_ERROR(err);
		err = clSetKernelArg(fc_kernel2, 1, sizeof(cl_mem), &buf1_1);    CHECK_ERROR(err);
		err = clEnqueueNDRangeKernel(kernel_queue, fc_kernel2, 2, NULL, global_size, local_size, 1, &kernel_event[0], &kernel_event[2]);	CHECK_ERROR(err);
		err = clEnqueueReadBuffer(write_queue, buf2_1, CL_FALSE, 0,
			sizeof(cl_float) * (batch_num * M), output2, 1, &kernel_event[2], &kernel_event[3]);	CHECK_ERROR(err);
	}

	clFinish(queue);
}
time_t startk, endk, startc, endc, startI, endI;

void cnn(float* images, float* network, int* labels, float* confidences, int num_of_image) {

	float* w[21];
	float* b[21];
	int offset = 0;
	// link weights and biases to network
	for (int i = 0; i < 17; ++i) {
		if (i == 2 || i == 5 || i == 9 || i == 13) i++;	// pooling layer has no weights and biases
		w[i] = network + offset;
		offset += 3 * 3 * INPUT_DIM[i] * OUTPUT_DIM[i];
		b[i] = network + offset;
		offset += OUTPUT_DIM[i];
	}
	for (int i = 18; i < 21; ++i) {
		w[i] = network + offset;
		offset += INPUT_DIM[i] * OUTPUT_DIM[i];
		b[i] = network + offset;
		offset += OUTPUT_DIM[i];
	}


	// allocate memory for layer
	float* layer[21];
	for (int i = 0; i < 21; ++i) {
		layer[i] = (float*)malloc(sizeof(float) * OUTPUT_DIM[i] * NBYN[i] * NBYN[i] * PARALLEL);
		//layer[i] = clEnqueueMapBuffer(kernel_queue, buf2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(float) * OUTPUT_DIM[i] * NBYN[i] * NBYN[i] * PARALLEL, 0, NULL, NULL, err);

		if (layer[i] == NULL) {
			perror("malloc error");
		}
	}


	// run network
	for (int i = 0; i < num_of_image; i += PARALLEL) {
		int j = 0;
		layer[j] = clEnqueueMapBuffer(kernel_queue, buf2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(float) * OUTPUT_DIM[j] * NBYN[j] * NBYN[j] * PARALLEL, 0, NULL, NULL, err);		CHECK_ERROR(err);
		clEnqueueUnmapMemObject(kernel_queue, buf2, layer[j++], 0, NULL, NULL);
		convolution_layer(images, layer[0], w[0], b[0], INPUT_DIM[0], OUTPUT_DIM[0], NBYN[0]);

		layer[j] = clEnqueueMapBuffer(kernel_queue, buf2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(float) * OUTPUT_DIM[j] * NBYN[j] * NBYN[j] * PARALLEL, 0, NULL, NULL, err);		CHECK_ERROR(err);
		clEnqueueUnmapMemObject(kernel_queue, buf2, layer[j++], 0, NULL, NULL);
		convolution_layer(layer[0], layer[1], w[1], b[1], INPUT_DIM[1], OUTPUT_DIM[1], NBYN[1]);

		layer[j] = clEnqueueMapBuffer(kernel_queue, buf2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(float) * OUTPUT_DIM[j] * NBYN[j] * NBYN[j] * PARALLEL, 0, NULL, NULL, err);		CHECK_ERROR(err);
		clEnqueueUnmapMemObject(kernel_queue, buf2, layer[j++], 0, NULL, NULL);
		pooling_layer(layer[1], layer[2], INPUT_DIM[2], NBYN[2]);

		layer[j] = clEnqueueMapBuffer(kernel_queue, buf2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(float) * OUTPUT_DIM[j] * NBYN[j] * NBYN[j] * PARALLEL, 0, NULL, NULL, err);		CHECK_ERROR(err);
		clEnqueueUnmapMemObject(kernel_queue, buf2, layer[j++], 0, NULL, NULL);
		convolution_layer(layer[2], layer[3], w[3], b[3], INPUT_DIM[3], OUTPUT_DIM[3], NBYN[3]);

		layer[j] = clEnqueueMapBuffer(kernel_queue, buf2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(float) * OUTPUT_DIM[j] * NBYN[j] * NBYN[j] * PARALLEL, 0, NULL, NULL, err);		CHECK_ERROR(err);
		clEnqueueUnmapMemObject(kernel_queue, buf2, layer[j++], 0, NULL, NULL);
		convolution_layer(layer[3], layer[4], w[4], b[4], INPUT_DIM[4], OUTPUT_DIM[4], NBYN[4]);

		layer[j] = clEnqueueMapBuffer(kernel_queue, buf2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(float) * OUTPUT_DIM[j] * NBYN[j] * NBYN[j] * PARALLEL, 0, NULL, NULL, err);		CHECK_ERROR(err);
		clEnqueueUnmapMemObject(kernel_queue, buf2, layer[j++], 0, NULL, NULL);
		pooling_layer(layer[4], layer[5], INPUT_DIM[5], NBYN[5]);

		layer[j] = clEnqueueMapBuffer(kernel_queue, buf2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(float) * OUTPUT_DIM[j] * NBYN[j] * NBYN[j] * PARALLEL, 0, NULL, NULL, err);		CHECK_ERROR(err);
		clEnqueueUnmapMemObject(kernel_queue, buf2, layer[j++], 0, NULL, NULL);
		convolution_layer(layer[5], layer[6], w[6], b[6], INPUT_DIM[6], OUTPUT_DIM[6], NBYN[6]);

		layer[j] = clEnqueueMapBuffer(kernel_queue, buf2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(float) * OUTPUT_DIM[j] * NBYN[j] * NBYN[j] * PARALLEL, 0, NULL, NULL, err);		CHECK_ERROR(err);
		clEnqueueUnmapMemObject(kernel_queue, buf2, layer[j++], 0, NULL, NULL);
		convolution_layer(layer[6], layer[7], w[7], b[7], INPUT_DIM[7], OUTPUT_DIM[7], NBYN[7]);

		layer[j] = clEnqueueMapBuffer(kernel_queue, buf2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(float) * OUTPUT_DIM[j] * NBYN[j] * NBYN[j] * PARALLEL, 0, NULL, NULL, err);		CHECK_ERROR(err);
		clEnqueueUnmapMemObject(kernel_queue, buf2, layer[j++], 0, NULL, NULL);
		convolution_layer(layer[7], layer[8], w[8], b[8], INPUT_DIM[8], OUTPUT_DIM[8], NBYN[8]);

		layer[j] = clEnqueueMapBuffer(kernel_queue, buf2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(float) * OUTPUT_DIM[j] * NBYN[j] * NBYN[j] * PARALLEL, 0, NULL, NULL, err);		CHECK_ERROR(err);
		clEnqueueUnmapMemObject(kernel_queue, buf2, layer[j++], 0, NULL, NULL);
		pooling_layer(layer[8], layer[9], INPUT_DIM[9], NBYN[9]);

		layer[j] = clEnqueueMapBuffer(kernel_queue, buf2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(float) * OUTPUT_DIM[j] * NBYN[j] * NBYN[j] * PARALLEL, 0, NULL, NULL, err);		CHECK_ERROR(err);
		clEnqueueUnmapMemObject(kernel_queue, buf2, layer[j++], 0, NULL, NULL);
		convolution_layer(layer[9], layer[10], w[10], b[10], INPUT_DIM[10], OUTPUT_DIM[10], NBYN[10]);

		layer[j] = clEnqueueMapBuffer(kernel_queue, buf2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(float) * OUTPUT_DIM[j] * NBYN[j] * NBYN[j] * PARALLEL, 0, NULL, NULL, err);		CHECK_ERROR(err);
		clEnqueueUnmapMemObject(kernel_queue, buf2, layer[j++], 0, NULL, NULL);
		convolution_layer(layer[10], layer[11], w[11], b[11], INPUT_DIM[11], OUTPUT_DIM[11], NBYN[11]);

		layer[j] = clEnqueueMapBuffer(kernel_queue, buf2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(float) * OUTPUT_DIM[j] * NBYN[j] * NBYN[j] * PARALLEL, 0, NULL, NULL, err);		CHECK_ERROR(err);
		clEnqueueUnmapMemObject(kernel_queue, buf2, layer[j++], 0, NULL, NULL);
		convolution_layer(layer[11], layer[12], w[12], b[12], INPUT_DIM[12], OUTPUT_DIM[12], NBYN[12]);

		layer[j] = clEnqueueMapBuffer(kernel_queue, buf2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(float) * OUTPUT_DIM[j] * NBYN[j] * NBYN[j] * PARALLEL, 0, NULL, NULL, err);		CHECK_ERROR(err);
		clEnqueueUnmapMemObject(kernel_queue, buf2, layer[j++], 0, NULL, NULL);
		pooling_layer(layer[12], layer[13], INPUT_DIM[13], NBYN[13]);

		layer[j] = clEnqueueMapBuffer(kernel_queue, buf2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(float) * OUTPUT_DIM[j] * NBYN[j] * NBYN[j] * PARALLEL, 0, NULL, NULL, err);		CHECK_ERROR(err);
		clEnqueueUnmapMemObject(kernel_queue, buf2, layer[j++], 0, NULL, NULL);
		convolution_layer(layer[13], layer[14], w[14], b[14], INPUT_DIM[14], OUTPUT_DIM[14], NBYN[14]);

		layer[j] = clEnqueueMapBuffer(kernel_queue, buf2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(float) * OUTPUT_DIM[j] * NBYN[j] * NBYN[j] * PARALLEL, 0, NULL, NULL, err);		CHECK_ERROR(err);
		clEnqueueUnmapMemObject(kernel_queue, buf2, layer[j++], 0, NULL, NULL);
		convolution_layer(layer[14], layer[15], w[15], b[15], INPUT_DIM[15], OUTPUT_DIM[15], NBYN[15]);

		layer[j] = clEnqueueMapBuffer(kernel_queue, buf2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(float) * OUTPUT_DIM[j] * NBYN[j] * NBYN[j] * PARALLEL, 0, NULL, NULL, err);		CHECK_ERROR(err);
		clEnqueueUnmapMemObject(kernel_queue, buf2, layer[j++], 0, NULL, NULL);
		convolution_layer(layer[15], layer[16], w[16], b[16], INPUT_DIM[16], OUTPUT_DIM[16], NBYN[16]);

		layer[j] = clEnqueueMapBuffer(kernel_queue, buf2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(float) * OUTPUT_DIM[j] * NBYN[j] * NBYN[j] * PARALLEL, 0, NULL, NULL, err);		CHECK_ERROR(err);
		clEnqueueUnmapMemObject(kernel_queue, buf2, layer[j++], 0, NULL, NULL);
		pooling_layer(layer[16], layer[17], INPUT_DIM[17], NBYN[17]);

		layer[j] = clEnqueueMapBuffer(kernel_queue, buf2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(float) * OUTPUT_DIM[j] * NBYN[j] * NBYN[j] * PARALLEL, 0, NULL, NULL, err);		CHECK_ERROR(err);
		clEnqueueUnmapMemObject(kernel_queue, buf2, layer[j++], 0, NULL, NULL);
		fc_layer(layer[17], layer[18], w[18], b[18], INPUT_DIM[18], OUTPUT_DIM[18]);

		layer[j] = clEnqueueMapBuffer(kernel_queue, buf2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(float) * OUTPUT_DIM[j] * NBYN[j] * NBYN[j] * PARALLEL, 0, NULL, NULL, err);		CHECK_ERROR(err);
		clEnqueueUnmapMemObject(kernel_queue, buf2, layer[j++], 0, NULL, NULL);
		fc_layer(layer[18], layer[19], w[19], b[19], INPUT_DIM[19], OUTPUT_DIM[19]);

		layer[j] = clEnqueueMapBuffer(kernel_queue, buf2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(float) * OUTPUT_DIM[j] * NBYN[j] * NBYN[j] * PARALLEL, 0, NULL, NULL, err);		CHECK_ERROR(err);
		clEnqueueUnmapMemObject(kernel_queue, buf2, layer[j++], 0, NULL, NULL);
		fc_layer(layer[19], layer[20], w[20], b[20], INPUT_DIM[20], OUTPUT_DIM[20]);


		float* result;
		for (int j = 0; j < PARALLEL; ++j) {
			result = layer[20] + 10 * j;
			softmax(result, 10);
			labels[i + j] = find_max(result, 10);
			confidences[i + j] = result[labels[i + j]];
		}
		images += 32 * 32 * 3 * PARALLEL;
	}


	//for (int i = 0; i < 21; ++i) {
	//	free(layer[i]);
	//}
}
