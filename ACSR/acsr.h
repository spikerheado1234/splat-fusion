#ifndef CUSTOM_DATA_STRUCTURE
#define CUSTOM_DATA_STRUCTURE

#include <utility> 

/*
 * This file contains the information of how the customised data-structure will look like 
 * in c++.
*/

/* Naive implementation over here, just to accelrate the writing of the code-gen scheme. */

typedef struct coordinate {
    int x, y;
} coordinate;

typedef struct vector {
    /*This represents the row-wise linear transformation: xi+y.*/
    float x,y;
} vector;

typedef struct vector_int {
    int x, y;
};

/* Note, these are all compile time constant and known ahead of time, 
* so we seaprate from our main data_structure to make it more lightweight.
* This way, repeated memcpys + mallocs will finish faster.
*/
typedef struct transformations {
        /* This gives the mapping from dense -> sparse. */
    vector * linear_transformation;
    vector_int * linear_transformation_spmm;

    // We have column_major ordered ACSRs. In this case, 
    //  it is not necessary that the size of the above two vectors
    //  are the third dimension.
    int size_transforms_arr; 

    // These are the dimensions of the output tensor.
    // Should usually correspond to: (b, n, s, sparsity_parameter).
    int dim_one, dim_two, dim_three, dim_four;

    int total_value_size;
    int max_width;
    int num_blocks;
    int block_height;

    // Number of non-zero values.
    int * nnzs;
    int * row_ptrs;

    // Additional meta-data
    int * col_start; // size ceil(Seq-length/T). -> T is number of threads.
    int * col_end;

    int * row_map; // row map for optimizing stride pattern
    int * block_stride;

    public:
        int get_size() {
            return this->dim_one * this->dim_two * this->dim_three * this->dim_four;
        }

} transformations;

//bool is_identical(data_structure *, transformations *, 
//                    int , int , int , 
//                    int , float);

transformations* generate_blocked_ds(int batch, int seq_length, int num_heads, int hidden_dimension, int sparsity_param);
transformations* generate_windowed_ds(int batch, int seq_length, int num_heads, int hidden_dimension, int sparsity_param);
transformations *generate_strided_ds(int batch, int seq_length, int num_heads, int hidden_dimension, int sparsity_param);
transformations * generate_strided_ds_micro(int batch, int seq_length, int num_heads, int hidden_dimension, int sparsity_param); 

void pretty_print(transformations* transforms);
void populate_column_metadata(transformations * transforms, int block_height);
int populate_strided_metadata(transformations* t, int block_size);

#endif
