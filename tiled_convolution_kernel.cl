 //globalsize: { D2 * D1 * N * N  , P}
 //localsize: { N*N, 1}
__kernel void convolution(__global float* inputs, __global float* outputs, __global float* filters, __local float* l_mem, int n, int d2) {
	int d1 = get_global_size(0) / (n * n * d2);

	int g_i = get_global_id(0);
	int page = get_global_id(1);
	int oc = g_i / (n * n * d1) % d2;
	int ic = g_i / (n * n) % d1;
	int img_i = g_i / n % n;
	int img_j = g_i % n;
	int l_i = get_local_id(0);

	int i_idx = (d1 * n * n * page) + n * n * ic;
	int o_idx = (d2 * n * n * d1 * page) + n * n * d1 * oc;
	int f_idx = 9 * (oc * d1 + ic);

	__local float filter[9];
	if (l_i < 9)
		filter[l_i] = filters[f_idx + l_i];
	l_mem[l_i] = inputs[i_idx + l_i];
	barrier(CLK_LOCAL_MEM_FENCE);

	float sum = 0;
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			int x = img_i + i - 1;
			int y = img_j + j - 1;
			int cur_img_idx = x * n + y;
			int cur_f_idx = i * 3 + j;
			if (x >= 0 && x < n && y >= 0 && y < n)
				sum += l_mem[cur_img_idx] * filter[cur_f_idx];
		}
	}
	outputs[o_idx + n * d1 * img_i + d1 * img_j + ic] = sum;	//[p][oc][i][j][ic]
}

//globalsize: { D2 * D1 * N * N  , P}
//localsize: { D1, 1}
__kernel void reduction(__global float* inputs, __global float* outputs, __constant float* biases, __local float* filterout, int n, int d2) {
	int d1 = get_global_size(0) / (n * n * d2);

	int g_i = get_global_id(0);
	int page = get_global_id(1);
	int oc = g_i / (d1 * n * n) % d2;

	int g_i = get_global_id(0);
	int img_i = g_i / n % n;
	int img_j = g_i % n;
	int l_i = get_local_id(0); // ic

	int o_idx = (d2 * n * n * d1 * page) + (n * n * d1 * oc) + n * d1 * img_i + d1 * img_j + l_i; // [p][oc][i][j][ic]

	filterout[l_i] = inputs[o_idx];
	//barrier(CLK_LOCAL_MEM_FENCE);

	
	//for (int p = d1 >> 1; p >= 1; p = p >> 1) {
	//	if (l_i < p) filterout[l_i] += filterout[l_i + p];
	//	barrier(CLK_LOCAL_MEM_FENCE);
	//}

	//if (l_i == 0) {
	//	float result = filterout[0];
	//	if (d1 == 3) result += filterout[2];
	//	result += biases[oc];
	//	output[o_idx] = (result > 0) ? result : 0;
	//}

	o_idx /= d1;
	int i = d1, next = d1 >> 2;
	while (next > 2) {
		if (l_i >= next) return;
		barrier(CLK_LOCAL_MEM_FENCE);
		for (int j = next; j < i; j += next) filterout[l_i] += filterout[l_i + j];
		i = next;
		next >>= 2;
	}
	barrier(CLK_LOCAL_MEM_FENCE);


	for (int j = 1; j < i; ++j) filterout[0] += filterout[j];
	filterout[0] += biases[oc];
	outputs[o_idx] = (filterout[0] > 0) ? filterout[0] : 0;
}