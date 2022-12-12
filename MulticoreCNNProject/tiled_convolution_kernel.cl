// globalsize: { P * D2, D1 * N * N }
// localsize: { 1, N * N }
__kernel void convolution(__global float *inputs, __global float *outputs, __global float *filters, __local float *l_mem, const int n, const int d2) {
	const int d1 = get_global_size(1) / (n * n);

	const int g_i = get_global_id(0);
	int page = g_i / d2;
	int oc = g_i % d2;

	const int g_j = get_global_id(1);
	int ic = g_i / (n * n);
	int img_i = g_j / n % n;
	int img_j = g_j % n;

	int i_idx = (d1 * n * n * page) + n * n * ic;
	int o_idx = (d2 * n * n * d1 * page) + n * n * d1 * oc;
	int f_idx = 9 * (oc * d1 + ic);

	l_mem[get_local_id(1)] = inputs[i_idx + img_i * n + img_j];
	barrier(CLK_LOCAL_MEM_FENCE);

	float sum = 0;
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			int x = img_i + i - 1;
			int y = img_j + j - 1;
			int cur_img_idx = x * n + y + i_idx;
			int cur_f_idx = i * 3 + j + f_idx;
			if (x >= 0 && x < n && y >= 0 && y < n)
				sum += l_mem[cur_img_idx] * filters[cur_f_idx];
		}
	}
	outputs[o_idx + n * d1 * img_i + d1 * img_j + get_group_id(1)] = sum;
}

// globalsize: { P * D2, D1 * N * N }
// localsize: { 1, D1 }
__kernel void reduction(__global float *inputs, __global float *outputs, __constant float *biases, __local float *filterout, const int n, const int d2) {
	const int d1 = get_global_size(1) / (n * n);

	const int g_i = get_global_id(0);
	int page = g_i / d2;
	int oc = g_i % d2;

	const int g_j = get_global_id(1);
	int img_i = g_j / n % n;
	int img_j = g_j % n;

	int l_j = get_local_id(1);
	int o_idx = (d2 * n * n * d1 * page) + (n * n * d1 * oc) + n * d1 * img_i + d1 * img_j + l_j;
	filterout[l_j] = inputs[o_idx];
	o_idx /= d1;

	int i = d1, next = d1 >> 2;
	while (next > 1) {
		if (l_j >= next) return;
		barrier(CLK_LOCAL_MEM_FENCE);
		for (int j = next; j < i; j += next) filterout[l_j] += filterout[l_j + j];
		i = next;
		next >>= 2;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int j = 1; j < i; ++j) filterout[0] += filterout[j];
	filterout[0] += biases[g_j];
	outputs[o_idx] = (filterout[0] > 0) ? filterout[0] : 0;
}