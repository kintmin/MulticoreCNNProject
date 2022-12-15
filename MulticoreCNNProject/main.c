#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cnn.h"
#include "compare.h"

const char* CLASS_NAME[] = {
	"airplane",
	"automobile",
	"bird",
	"cat",
	"deer",
	"dog",
	"frog",
	"horse",
	"ship",
	"truck"
};

void print_usage_and_exit(char** argv) {
	fprintf(stderr, "Usage: %s <number of image> <output>\n", argv[0]);
	fprintf(stderr, " e.g., %s 3000 result.out\n", argv[0]);
	exit(EXIT_FAILURE);
}

void* read_bytes(const char* fn, size_t n) {
	FILE* f = fopen(fn, "rb");
	void* bytes = malloc(n);
	size_t r = fread(bytes, 1, n, f);
	fclose(f);
	if (r != n) {
		fprintf(stderr,
			"%s: %zd bytes are expected, but %zd bytes are read.\n",
			fn, n, r);
		exit(EXIT_FAILURE);
	}
	return bytes;
}

/*
 * Read images from "cifar10_image.bin".
 * CIFAR-10 test dataset consists of 10000 images with (3, 32, 32) size.
 * Thus, 10000 * 3 * 32 * 32 * sizeof(float) = 122880000 bytes are expected.
 */
const int IMAGE_CHW = 3 * 32 * 32 * sizeof(float);
float* read_images(size_t n) {
	return (float*)read_bytes("cifar10_image.bin", n * IMAGE_CHW);
}

/*
 * Read labels from "cifar10_label.bin".
 * 10000 * sizeof(int) = 40000 bytes are expected.
 */
int* read_labels(size_t n) {
	return (int*)read_bytes("cifar10_label.bin", n * sizeof(int));
}

/*
 * Read network from "network.bin".
 * conv1_1 : weight ( 64,   3, 3, 3) bias ( 64)
 * conv1_2 : weight ( 64,  64, 3, 3) bias ( 64)
 * conv2_1 : weight (128,  64, 3, 3) bias (128)
 * conv2_2 : weight (128, 128, 3, 3) bias (128)
 * conv3_1 : weight (256, 128, 3, 3) bias (256)
 * conv3_2 : weight (256, 256, 3, 3) bias (256)
 * conv3_3 : weight (256, 256, 3, 3) bias (256)
 * conv4_1 : weight (512, 256, 3, 3) bias (512)
 * conv4_2 : weight (512, 512, 3, 3) bias (512)
 * conv4_3 : weight (512, 512, 3, 3) bias (512)
 * conv5_1 : weight (512, 512, 3, 3) bias (512)
 * conv5_2 : weight (512, 512, 3, 3) bias (512)
 * conv5_3 : weight (512, 512, 3, 3) bias (512)
 * fc1     : weight (512, 512) bias (512)
 * fc2     : weight (512, 512) bias (512)
 * fc3     : weight ( 10, 512) bias ( 10)
 * Thus, 60980520 bytes are expected.
 */
const int NETWORK_SIZES[] = {
	64 * 3 * 3 * 3, 64,
	64 * 64 * 3 * 3, 64,
	128 * 64 * 3 * 3, 128,
	128 * 128 * 3 * 3, 128,
	256 * 128 * 3 * 3, 256,
	256 * 256 * 3 * 3, 256,
	256 * 256 * 3 * 3, 256,
	512 * 256 * 3 * 3, 512,
	512 * 512 * 3 * 3, 512,
	512 * 512 * 3 * 3, 512,
	512 * 512 * 3 * 3, 512,
	512 * 512 * 3 * 3, 512,
	512 * 512 * 3 * 3, 512,
	512 * 512, 512,
	512 * 512, 512,
	10 * 512, 10
};

float* read_network() {
	return (float*)read_bytes("network.bin", 60980520);
}

float** slice_network(float* p) {
	float** r = (float**)malloc(sizeof(float*) * 32);
	for (int i = 0; i < 32; ++i) {
		r[i] = p;
		p += NETWORK_SIZES[i];
	}
	return r;
}

/*
	프로젝트 속성 - 구성속성 - 디버깅 - 명령인수
	꼭 '모든구성' 설정으로 되어있는지 봐주세요. (아니면 Release나 Debug만 설정될 수 있음)
	이미지는 총 10000장. 임시로 30장 넣었습니다.

	실행방법: cnn_opencl.c 또는 cnn_seq.c를 빌드에서 제외하고 실행

	main.c를 테스트하기 전에 sample.c를 실행해보고 openCL 기본설정이 잘 됐는지 확인해주세요.
		1. 프로젝트 속성 - 링커 - 일반 - 추가라이브러리 (기존의 것에 추가)
		2. 프로젝트 속성 - C/C++ - 일반 - 추가포함디렉토리 (기존의 것에 추가)
*/
int main(int argc, char** argv) {
	if (argc != 3) {
		print_usage_and_exit(argv);
	}

	int num_images = atoi(argv[1]);
	float* images = read_images(num_images);
	float* network = read_network();
	float** network_sliced = slice_network(network);
	int* labels = (int*)calloc(num_images, sizeof(int));
	float* confidences = (float*)calloc(num_images, sizeof(float));
	time_t start, end;

	printf("OpenCL_CNN\tImages: %4d\n", num_images);
	cnn_init();
	start = clock();
	cnn(images, network_sliced, labels, confidences, num_images);
	printf("\tExecution time: %f sec\n", (double)(clock() - start) / CLK_TCK);

	FILE* of = fopen(argv[2], "w");
	int* labels_ans = read_labels(num_images);
	double acc = 0;
	for (int i = 0; i < num_images; ++i) {
		fprintf(of, "Image %04d: %s %f\n", i, CLASS_NAME[labels[i]], confidences[i]);
		if (labels[i] == labels_ans[i]) ++acc;
	}
	fprintf(of, "Accuracy: %f\n", acc / num_images);
	fclose(of);

	printf("\tAccuracy: %f\n", acc / num_images);
	compare(argv[2]);

	free(images);
	free(network);
	free(network_sliced);
	free(labels);
	free(confidences);
	free(labels_ans);

	return 0;
}
