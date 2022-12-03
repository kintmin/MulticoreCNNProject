__kernel void convolution(__global float* inputs, __global float* filters, __local float* filterout, __global float* outputs, __global float* biases,
	int inputDim, int outputDim, int N) {
	int gi = get_global_id(1);
	int oc = gi / (N * N);	//output channel
	int ic = get_local_id(0);	//input channel && local id
	int j = gi % N;
	int i = (gi - oc * N * N - j) / N;	//global size {[oc][i][j],[ic]}를  설정함. 따라서 oc,i,j를 계산
	int k, l, x, y, p;
	float sum = 0;
	float* input = inputs + N * N * ic;
	float* output = outputs + N * N * oc;
	float* filter = filters + 3 * 3 * (oc * inputDim + ic);

	for (k = 0; k < 3; k++) {
		for (l = 0; l < 3; l++) {
			int x = i + k - 1;
			int y = j + l - 1;
			if (x >= 0 && x < N && y >= 0 && y < N)
				sum += (double) input[x * N + y] * filter[k * 3 + l];	//filter 계산한 후 local memory에 저장	
		}
	}

	filterout[ic] = sum;
	barrier(CLK_LOCAL_MEM_FENCE);

	for (p = get_local_size(0) / 2; p >= 1; p = p >> 1) {
		if (ic < p)
			filterout[ic] += filterout[ic + p];
		barrier(CLK_LOCAL_MEM_FENCE);
	} 

	float bias = biases[oc];
	if (ic == 0) {
		if (inputDim == 3)
			filterout[0] += filterout[2];
		output[i * N + j] = input[0];//(filterout[0] + bias) > 0 ? (filterout[0] + bias) : 0;
	}
}