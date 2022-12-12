// globalsize: { 9 * D1, P * D2 * N * N}
// localsize: { 9 * D1, 1 }
__kernel void convolution(__global float *inputs, __local float *filterout, __global float *outputs, __global float *biases, __global float* filters, 
	__private int n, __private int d2) {
	int l_s = get_local_size(0);
	int d1 = l_s / 9;
	int l_id = get_local_id(0);
	int ic = l_id / 9;
	int l = l_id % 3;
	int k = l_id / 3 % 3;
	int g_id = get_global_id(1);
	int bc = g_id / (d2 * n * n);
	int oc = g_id / (n * n) % d2;
	int i = g_id / n % n;
	int j = g_id % n;

	float* input = inputs + (d1 * n * n * bc) + n * n * ic;
	float* filter = filters + 3 * 3 * (oc * d1 + ic);

	int x = j + l - 1;
	int y = i + k - 1;
	int index = k * 3 + l;

	filterout[ic * 9 + index] =  (x >= 0 && x < n && y >= 0 && y < n) ? input[y * n + x] * filter[k * 3 + l] : 0;
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int p = 9 >> 1; p >= 1; p = p >> 1) {
		if (index < p) filterout[ic * 9 + index] += filterout[ic * 9 + index + p];
		barrier(CLK_LOCAL_MEM_FENCE);
	}	
	if (index == 0)
		filterout[ic * 9] += filterout[ic * 9 + 8];

	for (int p = d1 >> 1; p >= 1; p = p >> 1) {
		if (ic < p && index == 0) filterout[ic * 9] += filterout[(ic + p) * 9];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (l_id == 0) {
		float result = filterout[0];
		float* output = outputs + (d2 * n * n * bc) + n * n * oc;
		if (d1 == 3) result += filterout[2 * 9];
		result += biases[oc];
		result = (result > 0) ? result : 0;
		output[i*n+j] = result;
	}
}