// globalsize: { P * D1, D2 * N * N }
// localsize: { D1, 1 }
__kernel void convolution(__constant float *inputs, __constant float *filters, __local float *filterout, __global float *outputs, __constant float *biases, __constant int *ptr_n) {
	int n = *ptr_n;
	int d1 = get_local_size(0);
	int d2 = get_global_size(1) / (n * n);

	int l_id = get_local_id(0);	// ic (d1)
	int g_id = get_global_id(1);
	int g_i = get_global_id(0) / d1;	// parallel
	int g_j = g_id / (n * n);	// oc (d2)
	int g_k = g_id / n % n;	// i
	int g_l = g_id % n;	// j

	int i_idx = (d1 * n * n * g_i) + n * n * l_id;
	int o_idx = (d2 * n * n * g_i) + n * n * g_j;
	int f_idx = 3 * 3 * (g_j * d1 + l_id);

	float sum = 0;
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			int x = g_k + i - 1;
			int y = g_l + j - 1;
			int cur_i_idx = x * n + y + i_idx;
			int cur_f_idx = i * 3 + j + f_idx;
			if (x >= 0 && x < n && y >= 0 && y < n)
				sum += inputs[cur_i_idx] * filters[cur_f_idx];	// filter 계산한 후 local memory에 저장	
		}
	}
	filterout[l_id] = sum;

	int i = d1, next = d1 >> 2;
	while (next > 1) {
		if (l_id >= next) return;
		barrier(CLK_LOCAL_MEM_FENCE);
		for (int j = next; j < i; j += next) filterout[l_id] += filterout[l_id + j];
		i = next;
		next >>= 2;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int j = 1; j < i; ++j) filterout[0] += filterout[j];
	filterout[0] += biases[g_j];
	outputs[o_idx + g_k * n + g_l] = (filterout[0] > 0) ? filterout[0] : 0;
}
