#define TILE_NUM   1024
#define VEC_WIDTH    16

/*
	input: d1 * N * N
	tile: 1024
	vector: float16

	tile 미리 가져오고,
	각 tile마다 (3x3) x d2 계산을 실시.
	output: D2 * D1 * N * N (한 번에 16개씩 저장)

	globalsize: { P, D1 * N * N / 16 }
	localsize: { 1, 1024 / 16 }
	groupsize: { P, D1 * N * N / 1024 }
*/

__kernel void convolution(__constant float16 *inputs, __constant float *filters, __global float16 *outputs,
	const int n, const int d1, const int d2) {

	const int page = get_global_id(0);

	const int img_i = get_global_id(1);
	const int tile_i = get_local_id(1);
	const int ic = img_i * VEC_WIDTH / (n * n);

	__local float16 tile[64];	// get_local_size(1)
	tile[tile_i] = inputs[(page * d1 * n * n) + img_i];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int v = 0; v < VEC_WIDTH; ++v) {
		float cur_tile;
		switch (v) {
		case 0: cur_tile = tile[tile_i].s0; break;
		case 1: cur_tile = tile[tile_i].s1;  break;
		case 2: cur_tile = tile[tile_i].s2;  break;
		case 3: cur_tile = tile[tile_i].s3;  break;
		case 4: cur_tile = tile[tile_i].s4;  break;
		case 5: cur_tile = tile[tile_i].s5;  break;
		case 6: cur_tile = tile[tile_i].s6;  break;
		case 7: cur_tile = tile[tile_i].s7;  break;
		case 8: cur_tile = tile[tile_i].s8;  break;
		case 9: cur_tile = tile[tile_i].s9;  break;
		case 10: cur_tile = tile[tile_i].sA;  break;
		case 11: cur_tile = tile[tile_i].sB;  break;
		case 12: cur_tile = tile[tile_i].sC;  break;
		case 13: cur_tile = tile[tile_i].sD;  break;
		case 14: cur_tile = tile[tile_i].sE;  break;
		case 15: cur_tile = tile[tile_i].sF;  break;
		}

		for (int oc = 0; oc < d2; ++oc) {
			int f_idx = 9 * (d1 * oc + ic);
			float sum = 0;
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					int x = tile_i / d1 * VEC_WIDTH % n - 1;
					int y = tile_i / d1 * VEC_WIDTH / n - 1;
					if (x >= 0 && x < n && y >= 0 && y < n)
						sum += cur_tile * filters[f_idx + i * 3 + j];
				}
			}
			outputs[page * n * n * d2 + n * n * oc + n * img_i * n + img_j] += sum;
		}
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
}