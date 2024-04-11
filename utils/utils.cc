#include "utils.h"
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <cassert>
#include <cmath>
#include <utility>
#include <string.h>

void print_transformation(transformations* transforms, int sequence_length, int block_height) {
    std::cout << "dim 1:" << transforms->dim_one << std::endl;
    std::cout << "dim 2:" << transforms->dim_two << std::endl;
    std::cout << "dim 3:" << transforms->dim_three << std::endl;
    std::cout << "dim 4:" << transforms->dim_four << std::endl;
    std::cout << "max_width:" << transforms->max_width << std::endl;
    std::cout << "total_value_size:" << transforms->total_value_size << std::endl;
    std::cout << "number_blocks:" << transforms->num_blocks << std::endl;
    std::cout << "block_height:" << transforms->block_height << std::endl;
    for(int i = 0; i < sequence_length; i++) {
        std::cout << transforms->linear_transformation_spmm[i].x << ", " << transforms->linear_transformation_spmm[i].y << " : " << transforms->nnzs[i]<< std::endl;
    }
    if(block_height != -1) {
        for(int i = 0; i < std::ceil((float)sequence_length/block_height); i++) {
            std::cout << "col_start:" << transforms->col_start[i] << ", col_end: " << transforms->col_end[i] << std::endl;
        }
    }
}

void matrix_fill(float* matrix, int size) {
	std::srand((unsigned) std::time(NULL));
	for (int i = 0; i < size; i++) {
		// float rng = ((float)(std::rand()))  / ((float)(std::rand()));		
		float rng = ((float)(std::rand()))  / ((float)RAND_MAX);
		matrix[i] = rng;
	}
}

void print_matrix(float** matrix, int outer, int dim1, int dim2, int ld) {
	printf("[");
	for (int i = 0; i < outer; i++) {
		printf("[");
		for (int j = 0; j < dim1; j++) {
			printf("[");
			for (int k = 0; k < dim2; k++) {
				if (k < dim2-1) {
					printf("%.3f, ", matrix[i][j*ld+k]);
				} else {
					printf("%.3f");
				}
			} 
			if (j < dim1-2) {
				printf("]\n");
			} else {
				printf("]");
			}
		} 
		if (i < outer-1) {
			printf("]\n");
		} else {
			printf("]");
		}
	} 
	printf("]");
}

void print_matrix(int* matrix, int batch_size, int sequence_length, int num_heads, int hidden_dimension) {
	int size = batch_size * sequence_length * num_heads * hidden_dimension; 
#define idx(b, s, h, d) ((b*(sequence_length*num_heads*hidden_dimension)) + (s*num_heads*hidden_dimension) + (h*hidden_dimension) + d)
	printf("[");
	for (int b = 0; b < batch_size; b++) {
		printf("[");
		for (int s = 0; s < sequence_length; s++) {
			printf("[");
			for (int h = 0; h < num_heads; h++) {
				printf("[");
				for (int d = 0; d < hidden_dimension; d++) {
					// We print this element here.
					if (d < (hidden_dimension - 1)) {
						printf("%d, ", matrix[idx(b, s, h, d)]);
					} else {
						printf("%d", matrix[idx(b, s, h, d)]);
					}
				} 
				if (h < (num_heads - 1)) {
					printf("],\n");
				} else {
					printf("]");
				}
			}
			if (s < sequence_length - 1) {
				printf("],\n\n");
			} else {
				printf("]");
			}
		}
		if (b < batch_size - 1) {
			printf("],\n\n\n");
		} else {
			printf("]");
		}
	}
	printf("]");
}

void print_matrix(float* matrix, int batch_size, int sequence_length, int num_heads, int hidden_dimension) {
	int size = batch_size * sequence_length * num_heads * hidden_dimension; 
#define idx(b, s, h, d) ((b*(sequence_length*num_heads*hidden_dimension)) + (s*num_heads*hidden_dimension) + (h*hidden_dimension) + d)
	printf("[");
	for (int b = 0; b < batch_size; b++) {
		printf("[");
		for (int s = 0; s < sequence_length; s++) {
			printf("[");
			for (int h = 0; h < num_heads; h++) {
				printf("[");
				for (int d = 0; d < hidden_dimension; d++) {
					// We print this element here.
					if (d < (hidden_dimension - 1)) {
						printf("%.2f, ", matrix[idx(b, s, h, d)]);
					} else {
						printf("%.2f", matrix[idx(b, s, h, d)]);
					}
				} 
				if (h < (num_heads - 1)) {
					printf("],\n");
				} else {
					printf("]");
				}
			}
			if (s < sequence_length - 1) {
				printf("],\n\n");
			} else {
				printf("]");
			}
		}
		if (b < batch_size - 1) {
			printf("],\n\n\n");
		} else {
			printf("]");
		}
	}
	printf("]");
}

void transpose(const float *tensor, size_t d0, size_t d1, size_t d2, size_t d3,
		float *output, size_t dim1, size_t dim2) {
	assert(dim1 < 4 && dim2 < 4);
	for (size_t i = 0; i < d0; i++) {
		for (size_t j = 0; j < d1; j++) {
			for (size_t k = 0; k < d2; k++) {
				for (size_t x = 0; x < d3; x++) {
					size_t idx[] = {i, j, k, x};
					size_t len[] = {d0, d1, d2, d3};
					std::swap(idx[dim1], idx[dim2]);
					std::swap(len[dim1], len[dim2]);
					size_t inp_idx = ((((i * d1) + j) * d2) + k) * d3 + x;
					size_t out_idx = ((((idx[0] * len[1]) + idx[1]) * len[2]) + idx[2]) * len[3] + idx[3];
					output[out_idx] = tensor[inp_idx];
				}
			}
		}
	}
}

void fill_rand(float *A, size_t size) {
	std::random_device rd;
	std::mt19937 rng(rd());
	std::uniform_real_distribution<float> dist;
	for (size_t i = 0; i < size; i++) {
		A[i] = dist(rng);
	}
}

void fill_mask_column(float *mask, int s, int d) {
	int stride = d + 1;
	std::fill(mask, mask + s * s, 0);
	for (int i = 0; i < s; i++) {
		if (i % stride == 0) {
			for (int j = i; j < s; j++) {
				mask[i * s + j] = 1;
				mask[j * s + i] = 1;
			}
		}
	}
}

void fill_mask_windowed(float * mask, int s, int w) {
	std::fill(mask, mask + s*s, 0);
	for (int i = 0; i < s; i++) {
		for (int j = 0; j < s; j++) {
			if ((i - w) <= j && j <= (i+w)) {
				mask[i*s+j] = 1;
			}
		}
	}
}

void fill_mask_strided(float * mask, int s, int w) {
	std::fill(mask, mask + s*s, 0);
	for (int i = 0; i < s; i++) {
		for (int j = 0; j < s; j++) {
			if (abs(j-i)%w == 0) {
				mask[i*s+j] = 1;
			}
		}
	}
}

void fill_mask_blocked(float * mask, int s, int w) {
	std::fill(mask, mask + s*s, 0);
	for (int i = 0; i < s; i++) {
		for (int j = 0; j < s; j++) {
			// We have several cases.
			// case 1: i is in block num of 0.
			int block_num = i / w;
			if (block_num == 0) {
				if (j < w) {
					mask[i*s+j] = 1;
				}
			} else {
				if (j >= ((block_num-1)*w) && (j < ((block_num-1)*w + 2*w))) {
					mask[i*s+j] = 1;
				}
			}
		}
	}
}

bool allclose(const float *A, const float *B, size_t size, float tolerance) {
	for (size_t i = 0; i < size; i++) {
		if (std::fabs(A[i] - B[i]) > tolerance) {
			return false;
		}
	}
	return true;
}


void batch_sddmm(const float *A, const float *B, const float *mask,
		size_t d0, size_t d1, size_t d2, size_t d3, float *C) {
#define _A(_i, _j, _r, _c) \
	A[(((((((_i) * d1) + (_j)) * d2) + (_r)) * d3) + (_c))]
#define _B(_i, _j, _r, _c) \
	B[(((((((_i) * d1) + (_j)) * d3) + (_r)) * d2) + (_c))]
#define _C(_i, _j, _r, _c) \
	C[(((((((_i) * d1) + (_j)) * d2) + (_r)) * d2) + (_c))]
#define _M(_i, _j) \
	mask[(_i) * d2 + (_j)]
	for (size_t i = 0; i < d0; i++) {
		for (size_t j = 0; j < d1; j++) {
			for (size_t row = 0; row < d2; row++) {
				for (size_t col = 0; col < d2; col++) {
					float sum = 0.f;
					if (_M(row, col) != 0) {
						for (size_t k = 0; k < d3; k++) {
							sum += _A(i, j, row, k) * _B(i, j, k, col);
						}
					}
					_C(i, j, row, col) = sum;
				}
			}
		}
	}
#undef _M
#undef _C
#undef _B
#undef _A
}

size_t dense2csr_batched(const float * A, size_t d0, size_t d1, size_t batch_dim, int **ind, int **col, float**val) {
	std::vector<int> csr_ind;
	std::vector<int> csr_col;
	std::vector<float> csr_val;
	csr_ind.push_back(0);

	for (int batch = 0; batch < batch_dim; batch++) {
		for (int i = 0; i < d0; i++) {
			int count = 0;
			for (int j = 0; j < d1; j++) {
				float val_temp = A[(i * d1) + j];
				if (val_temp != 0) {
					csr_val.push_back(0.0f);
					csr_col.push_back(j);
					count++;
				}
			}
			if (batch == 0) {
				csr_ind.push_back(csr_ind.back() + count);
			}
		}
	}

	*ind = new int[csr_ind.size()];
	*col = new int[csr_col.size()];
	*val = new float[csr_val.size()];
	std::copy(csr_ind.begin(), csr_ind.end(), *ind);
	// We replicate the column, indexes again. For batch number of times //
	std::copy(csr_col.begin(), csr_col.end(), *col);
	std::copy(csr_val.begin(), csr_val.end(), *val);

	return csr_val.size();
}

size_t dense2csr(const float *A, size_t d0, size_t d1, int **ind, int **col, float **val) {
	std::vector<int> csr_ind;
	std::vector<int> csr_col;
	std::vector<float> csr_val;
	csr_ind.push_back(0);

	for (int i = 0; i < d0; i++) {
		int count = 0;
		for (int j = 0; j < d1; j++) {
			float val = A[i * d1 + j];
			if (val != 0) {
				csr_col.push_back(j);
				csr_val.push_back(val);
				count++;
			}
		}
		csr_ind.push_back(csr_ind.back() + count);
	}

	*ind = new int[csr_ind.size()];
	*col = new int[csr_col.size()];
	*val = new float[csr_val.size()];
	std::copy(csr_ind.begin(), csr_ind.end(), *ind);
	std::copy(csr_col.begin(), csr_col.end(), *col);
	std::copy(csr_val.begin(), csr_val.end(), *val);

	return csr_val.size();
}

// Converts mat_one, a one-d tensor representation of a 4-d tensor of size: [b, n, s, h]
// into a two-d tesnor of size: [b*s, n*h].
void un_flatten(float * mat_one, float *** mat_two, size_t batch, size_t seq, size_t heads, size_t hid_dim) {
	*mat_two = (float**)malloc(sizeof(float*)*batch*heads);
	for (int i = 0; i < batch*heads; i++) {
		// First allocate the relevant memory in mat_two.
		*mat_two[i] = (float*)malloc(sizeof(float)*seq*hid_dim);
		memcpy(*mat_two[i], &mat_one[i*batch*heads*seq*hid_dim], sizeof(float)*seq*hid_dim);	
	}	
}