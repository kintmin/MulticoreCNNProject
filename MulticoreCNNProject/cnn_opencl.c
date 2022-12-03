#include <CL/cl.h>
#include "cnn.h"
#pragma warning(disable : 4996)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

cl_int err;
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue kernel_queue;
size_t kernel_source_size;
cl_program program;

#define CHECK_ERROR(err) \
    if(err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

char* get_source_code(const char* file_name, size_t* len) {
    FILE* file = fopen(file_name, "rb");
    if (file == NULL) {
        printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    size_t length = (size_t)ftell(file);
    rewind(file);

    char* source_code = (char*)malloc(length + 1);
    fread(source_code, length, 1, file);
    source_code[length] = '\0';
    fclose(file);
    *len = length;

    return source_code;
}

void build_error(cl_program program, cl_device_id device, cl_int err) {
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        char* log;

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        CHECK_ERROR(err);

        log = (char*)malloc(log_size + 1);
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        CHECK_ERROR(err);

        log[log_size] = '\0';
        printf("Compiler error:\n%s\n", log);
        free(log);
        exit(0);
    };
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

/*
 * TODO
 * Define global variables here. For example,
 * cl_platform_id platform;
 */

void ConvolutionLayer(float* input, float* output, float* filter, float* biases, int outputDim, int inputDim, int N) {
    cl_kernel convolution = clCreateKernel(program, "convolution", &err); CHECK_ERROR(err);
    //cl_kernel integral2 = clCreateKernel(program, "convolution", &err); CHECK_ERROR(err);
    printf(" %p ", input);
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * inputDim * N * N, input, &err); CHECK_ERROR(err);
    cl_mem filterBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * outputDim * inputDim * 3 * 3, filter, &err); CHECK_ERROR(err);
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * outputDim * N * N, NULL, &err); CHECK_ERROR(err);
    cl_mem biasesBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * outputDim, biases, &err); CHECK_ERROR(err);
    //cl_mem outputBuffer2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * globalsize / localsize, NULL, &err); CHECK_ERROR(err);

    err = clSetKernelArg(convolution, 0, sizeof(cl_mem), &inputBuffer); CHECK_ERROR(err);
    err = clSetKernelArg(convolution, 1, sizeof(cl_mem), &filterBuffer); CHECK_ERROR(err);
    err = clSetKernelArg(convolution, 2, sizeof(float) * inputDim, NULL); CHECK_ERROR(err);
    err = clSetKernelArg(convolution, 3, sizeof(cl_mem), &outputBuffer); CHECK_ERROR(err);
    err = clSetKernelArg(convolution, 4, sizeof(cl_mem), &biasesBuffer); CHECK_ERROR(err);
    err = clSetKernelArg(convolution, 5, sizeof(int), &inputDim); CHECK_ERROR(err);
    err = clSetKernelArg(convolution, 6, sizeof(int), &outputDim); CHECK_ERROR(err);
    err = clSetKernelArg(convolution, 7, sizeof(int), &N); CHECK_ERROR(err);
    /*err = clSetKernelArg(integral2, 0, sizeof(cl_mem), &outputBuffer2); CHECK_ERROR(err);
    err = clSetKernelArg(integral2, 1, sizeof(double) * localsize, NULL); CHECK_ERROR(err);
    err = clSetKernelArg(integral2, 2, sizeof(int), &N); CHECK_ERROR(err);
    err = clSetKernelArg(integral2, 3, sizeof(int), &initialNum2); CHECK_ERROR(err);*/

    size_t global_size[2] = {inputDim, N * N * outputDim };
    size_t local_size[2] = { inputDim, 1};
    cl_event kernel_event[4];
    clEnqueueNDRangeKernel(kernel_queue, convolution, 2, NULL, global_size, local_size, 0, NULL, NULL);
    err = clEnqueueReadBuffer(kernel_queue, outputBuffer, CL_TRUE, 0, sizeof(float) * outputDim * N * N, output, 0, NULL, NULL);
    //clEnqueueNDRangeKernel(kernel_queue, integral2, 1, NULL, &global_size, &local_size, 1, &kernel_event[0], &kernel_event[2]);
    //err = clEnqueueReadBuffer(write_queue, outputBuffer2, CL_FALSE, 0, sizeof(double) * globalsize / localsize, sum2, 1, &kernel_event[2], &kernel_event[3]);

    //clFinish(kernel_queue);
}


static void pooling2x2(float* input, float* output, int N) {
    int i, j, k, l;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            float max = 0;
            for (k = 0; k < 2; k++) {
                for (l = 0; l < 2; l++) {
                    float pixel = input[(i * 2 + k) * 2 * N + j * 2 + l];
                    max = (max > pixel) ? max : pixel;
                }
            }
            output[i * N + j] = max;
        }
    }
}

/*
 * D = channel size
 * N = width and height of an output image
 * Thus, input is (D, N * 2, N * 2) and output is (D, N, N).
 */
static void pooling_layer(float* inputs, float* outputs, int D, int N) {
    int i;
    for (i = 0; i < D; i++) {
        float* input = inputs + i * N * N * 4;
        float* output = outputs + i * N * N;
        pooling2x2(input, output, N);
    }
}

/*
 * D2 = output channel size
 * D1 = input channel size
 * N = width and height of an input image
 * input image is zero-padded by 1.
 * Thus, input is (D1, N, N) and output is (D2, N, N)
 */
#define ReLU(x) (((x)>0)?(x):0)

 /*
  * M = output size
  * N = input size
  */
static void fc_layer(float* input_neuron, float* output_neuron, float* weights, float* biases, int M, int N) {
    int i, j;
    for (j = 0; j < M; j++) {
        float sum = 0;
        for (i = 0; i < N; i++) {
            sum += input_neuron[i] * weights[j * N + i];
        }
        sum += biases[j];
        output_neuron[j] = ReLU(sum);
    }
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

void cnn_init() {
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err);


    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
    CHECK_ERROR(err);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);

    kernel_queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
    CHECK_ERROR(err);

    char* kernel_source = get_source_code("convolution_2.cl", &kernel_source_size);
    program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
    CHECK_ERROR(err);

    err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    build_error(program, device, err);
    CHECK_ERROR(err);
}

static void convolution3x3(float* input, float* output, float* filter, int N) {
    int i, j, k, l;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            float sum = 0;
            for (k = 0; k < 3; k++) {
                for (l = 0; l < 3; l++) {
                    int x = i + k - 1;
                    int y = j + l - 1;
                    if (x >= 0 && x < N && y >= 0 && y < N)
                        sum += input[x * N + y] * filter[k * 3 + l];
                }
            }
            output[i * N + j] += sum;
        }
    }
}

/*
 * D2 = output channel size
 * D1 = input channel size
 * N = width and height of an input image
 * input image is zero-padded by 1.
 * Thus, input is (D1, N, N) and output is (D2, N, N)
 */
#define ReLU(x) (((x)>0)?(x):0)
static void convolution_layer(float* inputs, float* outputs, float* filters, float* biases, int D2, int D1, int N) {
    int i, j;

    memset(outputs, 0, sizeof(float) * N * N * D2);

    for (j = 0; j < D2; j++) {
        for (i = 0; i < D1; i++) {
            float* input = inputs + N * N * i;
            float* output = outputs + N * N * j;
            float* filter = filters + 3 * 3 * (j * D1 + i);
            convolution3x3(input, output, filter, N);
        }
    }

    for (i = 0; i < D2; i++) {
        float* output = outputs + N * N * i;
        float bias = biases[i];
        for (j = 0; j < N * N; j++) {
            output[j] = ReLU(output[j] + bias);
        }
    }
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
    c1_1 = alloc_layer(64 * 32 * 32);
    c1_2 = alloc_layer(64 * 32 * 32);
    p1 = alloc_layer(64 * 16 * 16);
    c2_1 = alloc_layer(128 * 16 * 16);
    c2_2 = alloc_layer(128 * 16 * 16);
    p2 = alloc_layer(128 * 8 * 8);
    c3_1 = alloc_layer(256 * 8 * 8);
    c3_2 = alloc_layer(256 * 8 * 8);
    c3_3 = alloc_layer(256 * 8 * 8);
    p3 = alloc_layer(256 * 4 * 4);
    c4_1 = alloc_layer(512 * 4 * 4);
    c4_2 = alloc_layer(512 * 4 * 4);
    c4_3 = alloc_layer(512 * 4 * 4);
    p4 = alloc_layer(512 * 2 * 2);
    c5_1 = alloc_layer(512 * 2 * 2);
    c5_2 = alloc_layer(512 * 2 * 2);
    c5_3 = alloc_layer(512 * 2 * 2);
    p5 = alloc_layer(512 * 1 * 1);
    fc1 = alloc_layer(512);
    fc2 = alloc_layer(512);
    fc3 = alloc_layer(10);

    float* temp = alloc_layer(64 * 32 * 32);

    // run network
    for (int i = 0; i < num_images; ++i)
    {
        float* image = images + i * 3 * 32 * 32;
        printf("\n%p !", image);
        memset(c1_1, 0, sizeof(float) * 64 * 32 * 32);


        ConvolutionLayer(image, c1_1, w1_1, b1_1, 64, 3, 32);
		/*image = images + i * 3 * 32 * 32;
		convolution_layer(image, temp, w1_1, b1_1, 64, 3, 32);
		int k = 0;
		float dif = 0;
		if (i == 24) {
			for (int j = 0; j < 2000; j++) {
				if (c1_1[j] != temp[j]) {
					printf("%d %f %f\n", j, c1_1[j], temp[j]);
				}
			}
		}*/
        printf(" %f %f\n", c1_1[0], image[0]);
        ConvolutionLayer(c1_1, c1_2, w1_2, b1_2, 64, 64, 32);
        pooling_layer(c1_2, p1, 64, 16);

        ConvolutionLayer(p1, c2_1, w2_1, b2_1, 128, 64, 16);
        ConvolutionLayer(c2_1, c2_2, w2_2, b2_2, 128, 128, 16);
        pooling_layer(c2_2, p2, 128, 8);

        ConvolutionLayer(p2, c3_1, w3_1, b3_1, 256, 128, 8);
        ConvolutionLayer(c3_1, c3_2, w3_2, b3_2, 256, 256, 8);
        ConvolutionLayer(c3_2, c3_3, w3_3, b3_3, 256, 256, 8);
        pooling_layer(c3_3, p3, 256, 4);

        ConvolutionLayer(p3, c4_1, w4_1, b4_1, 512, 256, 4);
        ConvolutionLayer(c4_1, c4_2, w4_2, b4_2, 512, 512, 4);
        ConvolutionLayer(c4_2, c4_3, w4_3, b4_3, 512, 512, 4);
        pooling_layer(c4_3, p4, 512, 2);

        ConvolutionLayer(p4, c5_1, w5_1, b5_1, 512, 512, 2);
        ConvolutionLayer(c5_1, c5_2, w5_2, b5_2, 512, 512, 2);
        ConvolutionLayer(c5_2, c5_3, w5_3, b5_3, 512, 512, 2);
        pooling_layer(c5_3, p5, 512, 1);

        //convolution_layer(image, c1_1, w1_1, b1_1, 64, 3, 32);
        //convolution_layer(c1_1, c1_2, w1_2, b1_2, 64, 64, 32);
        //pooling_layer(c1_2, p1, 64, 16);

        //convolution_layer(p1, c2_1, w2_1, b2_1, 128, 64, 16);
        //convolution_layer(c2_1, c2_2, w2_2, b2_2, 128, 128, 16);
        //pooling_layer(c2_2, p2, 128, 8);

        //convolution_layer(p2, c3_1, w3_1, b3_1, 256, 128, 8);
        //convolution_layer(c3_1, c3_2, w3_2, b3_2, 256, 256, 8);
        //convolution_layer(c3_2, c3_3, w3_3, b3_3, 256, 256, 8);
        //pooling_layer(c3_3, p3, 256, 4);

        //convolution_layer(p3, c4_1, w4_1, b4_1, 512, 256, 4);
        //convolution_layer(c4_1, c4_2, w4_2, b4_2, 512, 512, 4);
        //convolution_layer(c4_2, c4_3, w4_3, b4_3, 512, 512, 4);
        //pooling_layer(c4_3, p4, 512, 2);

        //convolution_layer(p4, c5_1, w5_1, b5_1, 512, 512, 2);
        //convolution_layer(c5_1, c5_2, w5_2, b5_2, 512, 512, 2);
        //convolution_layer(c5_2, c5_3, w5_3, b5_3, 512, 512, 2);
        //pooling_layer(c5_3, p5, 512, 1);

        fc_layer(p5, fc1, w1, b1, 512, 512);
        fc_layer(fc1, fc2, w2, b2, 512, 512);
        fc_layer(fc2, fc3, w3, b3, 10, 512);

        
        softmax(fc3, 10);

        labels[i] = find_max(fc3, 10);
        confidences[i] = fc3[labels[i]];
    }

    free(c1_1); free(c1_2); free(p1);
    free(c2_1); free(c2_2); free(p2);
    free(c3_1); free(c3_2); free(c3_3); free(p3);
    free(c4_1); free(c4_2); free(c4_3); free(p4);
    free(c5_1); free(c5_2); free(c5_3); free(p5);
    free(fc1); free(fc2); free(fc3);
} 