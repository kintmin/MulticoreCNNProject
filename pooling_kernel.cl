// global: { P, D * N * N }
__kernel void pooling(__global float *inputs, __global float *outputs, __private int n) {
	int g_id = get_global_id(1);
	int d = get_global_size(1) / (n * n);
	int g_i = get_global_id(0);	// PARALLEL
	int g_j = g_id / (n * n);
	int g_k = g_id / n % n;
	int g_l = g_id % n;

	float *input = inputs + (g_i * d * n * n * 4) + (g_j * n * n * 4);
	float *output = outputs + (g_i * d * n * n) + (g_j * n * n);

	float max = 0, pixel;
	int i_idx = ((g_k * 2) * (2 * n)) + (g_l * 2);
	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 2; ++j) {
			pixel = input[((g_k * 2 + i) * (2 * n)) + (g_l * 2 + j)];
			max = (max > pixel) ? max : pixel;
		}
	}

	output[g_k * n + g_l] = max;
}