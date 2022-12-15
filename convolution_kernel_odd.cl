// globalsize: { P * D1, D2 * N * N }
// localsize: { D1, 1 }
__kernel void convolution(__global float *inputs, __global float *filters, __local float *filterout, __global float *outputs, __constant float *biases, __private int n) {
	int d1 = get_local_size(0);
	int d2 = get_global_size(1) / (n * n);

	int l_id = get_local_id(0);	// ic (d1)
	int g_id = get_global_id(1);
	int g_i = get_global_id(0) / d1;	// parallel
	int g_j = g_id / (n * n);	// oc (d2)
	int g_k = g_id / n % n;	// i
	int g_l = g_id % n;	// j

	float *input = inputs + (d1 * n * n * g_i) + n * n * l_id;
	float *output = outputs + (d2 * n * n * g_i) + n * n * g_j;
	float *filter = filters + 3 * 3 * (g_j * d1 + l_id);

	float sum = 0;
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			int x = g_k + i - 1;
			int y = g_l + j - 1;
			if (x >= 0 && x < n && y >= 0 && y < n)
				sum += input[x * n + y] * filter[i * 3 + j];	// filter 계산한 후 local memory에 저장	
		}
	}

	filterout[l_id] = sum;
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int p = d1 >> 1; p >= 1; p = p >> 1) {
		if (l_id < p) filterout[l_id] += filterout[l_id + p];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (l_id == 0) {
		if (d1 == 3) filterout[0] += filterout[2];
		filterout[0] += biases[g_j];
		output[g_k * n + g_l] = (filterout[0] > 0) ? filterout[0] : 0;
	}
}