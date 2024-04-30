#include "acsr.h"
#include <cmath>
#include <stdexcept>
#include <cstdlib>
#include <assert.h>
#include <string>
#include <iostream>
#include <algorithm>
#include <map>
#include "../utils/utils.h"

// Eigen is a c++ fast library for numerical algorithms.
#include <Eigen/Dense>

void populate_column_metadata(transformations *transforms, int block_height)
{
    int total_height = transforms->dim_three;

    int num_blocks = ceil(float(total_height) / float(block_height));
    transforms->col_start = (int *)malloc(sizeof(int) * num_blocks);
    transforms->col_end = (int *)malloc(sizeof(int) * num_blocks);
    transforms->block_height = block_height;
    transforms->num_blocks = num_blocks;

    for (int block_num = 0; block_num < num_blocks; block_num++)
    {
        int col_start_idx = transforms->dim_three;
        int col_end_idx = 0;

        for (int i = block_num * block_height; i < (block_num + 1) * block_height; i++)
        {
            if (i < transforms->dim_three)
            {
                // Then we have to switch up col_start and col_end.

                // Unwrap the true column.
                int dense_y_col_start = round((float(0) - transforms->linear_transformation[i].y) * float(float(1) / transforms->linear_transformation[i].x));
                int dense_y_col_end = round((float(transforms->nnzs[i] - 1) - transforms->linear_transformation[i].y) * float(float(1) / transforms->linear_transformation[i].x));

                col_start_idx = std::min(col_start_idx, dense_y_col_start);
                col_end_idx = std::max(col_end_idx, dense_y_col_end);
            }
        }

        if (col_start_idx > col_end_idx)
        {
            throw std::invalid_argument("Issue occured when resolving col start/end metadata.\n");
        }

        transforms->col_start[block_num] = col_start_idx;
        transforms->col_end[block_num] = col_end_idx;
    }
}

int populate_strided_metadata(transformations *t, int block_size)
{
    int sequence_length = t->dim_three;
    std::map<std::pair<int, int>, std::vector<vector_int *>> groups;
    for (int i = 0; i < sequence_length; i++)
    {
        vector_int *_row = &t->linear_transformation_spmm[i];
        groups[{(*_row).x, (*_row).y % (*_row).x}].push_back(_row);
    }

    int new_index = 0;
    int groupNumber = 1;
    for (auto &[key, groupItems] : groups)
    {
        for (auto itemPtr : groupItems)
        {
            new_index++;
        }
        if (new_index % block_size != 0)
        {
            int padded = (new_index / block_size + 1) * block_size;
            new_index = padded;
        }
        groupNumber++;
    }
    t->row_map = new int[new_index];

    new_index = 0;
    for (auto &[key, groupItems] : groups)
    {
        for (auto itemPtr : groupItems)
        {
            int index = (itemPtr - t->linear_transformation_spmm);
            t->row_map[new_index] = index;
            new_index++;
        }
        if (new_index % block_size != 0)
        {
            int padded = (new_index / block_size + 1) * block_size;
            for (int i = new_index; i < padded; i++)
            {
                t->row_map[i] = -1;
            }
            new_index = padded;
        }
    }

    // populate the transformation
    t->num_blocks = new_index / block_size;
    t->block_height = block_size;

    t->col_start = new int[t->num_blocks];
    t->col_end = new int[t->num_blocks];
    t->block_stride = new int[t->num_blocks];
    for (int i = 0; i < t->num_blocks; i++)
    {
        t->col_start[i] = sequence_length;
        t->col_end[i] = 0;
        for (int j = 0; j < block_size; j++)
        {
            int index = i * block_size + j;
            int t_index = t->row_map[index];
            if (t->row_map[index] == -1)
            {
                break;
            }
            t->block_stride[i] = t->linear_transformation_spmm[t_index].x;
            t->col_start[i] = std::min(t->col_start[i], t->linear_transformation_spmm[t_index].y);
            t->col_end[i] = std::max(t->col_end[i], t->linear_transformation_spmm[t_index].y +
                                                        t->nnzs[t_index] * t->linear_transformation_spmm[t_index].x);
        }
    }
    return t->num_blocks * block_size;
}

transformations * generate_strided_ds_micro(int batch, int seq_length,
                                     int num_heads, int hidden_dimension, int sparsity_param) 
{

    // Window width is: 2*sparsity_param + 1.
    int dim_one = batch;
    int dim_two = num_heads;
    int dim_three = seq_length;
    int dim_four = (int)ceil(((float)seq_length) / ((float)sparsity_param));
    int size_transforms_arr = seq_length;

    size_t tensor_size = dim_one * dim_two * dim_three * dim_four;

    // First allocate them on the heap.
    //   Set the necessary values within the transformations struct.
    transformations *transform = (transformations *)malloc(sizeof(transformations));
    transform->linear_transformation = (vector *)malloc(sizeof(vector) * dim_three);
    transform->linear_transformation_spmm = (vector_int *)malloc(sizeof(vector_int) * dim_three);
    transform->nnzs = (int *)malloc(sizeof(int) * dim_three);
    transform->dim_one = dim_one;
    transform->dim_two = dim_two;
    transform->dim_three = dim_three;
    transform->dim_four = dim_four;
    transform->total_value_size = transform->dim_three * transform->dim_four;
    transform->max_width = transform->dim_four;
    transform->size_transforms_arr = size_transforms_arr;

    for (int i = 0; i < dim_three; i++)
    {

        int dense_one = i % sparsity_param;
        int dense_two = dense_one + sparsity_param;

        if (dense_two >= seq_length) {
            // We need to special case this over here.
            vector current_transform = {1, -dense_one};
            vector_int transform_spmm = {int(round(1 / current_transform.x)), int(round(-1 * current_transform.y))};

            int nnzs = 1;
            transform->linear_transformation[i] = current_transform;
            transform->linear_transformation_spmm[i] = transform_spmm;
            transform->nnzs[i] = nnzs;
        }

        // We can directly compute the metadata here. Because strided is very easy.
        vector current_transform = {((float)1) / ((float)sparsity_param), -(i%sparsity_param)};
        vector_int transform_spmm = {int(round(1 / current_transform.x)), int(round(-1 * current_transform.y))};

        int nnzs = ceil(((float)(seq_length - dense_one)) / (float)sparsity_param);
        transform->linear_transformation[i] = current_transform;
        transform->linear_transformation_spmm[i] = transform_spmm;
        transform->nnzs[i] = nnzs;
    }

    return transform;
}

transformations *generate_strided_ds(int batch, int seq_length,
                                     int num_heads, int hidden_dimension, int sparsity_param)
{

    assert(seq_length >= sparsity_param && "Sequence length must be at least sparsity_parameter size!");

    // Window width is: 2*sparsity_param + 1.
    int dim_one = batch;
    int dim_two = num_heads;
    int dim_three = seq_length;
    int dim_four = (int)ceil(((float)seq_length) / ((float)sparsity_param));
    int size_transforms_arr = seq_length;

    size_t tensor_size = dim_one * dim_two * dim_three * dim_four;

    // First allocate them on the heap.
    //   Set the necessary values within the transformations struct.
    transformations *transform = (transformations *)malloc(sizeof(transformations));
    transform->linear_transformation = (vector *)malloc(sizeof(vector) * dim_three);
    transform->linear_transformation_spmm = (vector_int *)malloc(sizeof(vector_int) * dim_three);
    transform->nnzs = (int *)malloc(sizeof(int) * dim_three);
    transform->dim_one = dim_one;
    transform->dim_two = dim_two;
    transform->dim_three = dim_three;
    transform->dim_four = dim_four;
    transform->total_value_size = transform->dim_three * transform->dim_four;
    transform->max_width = transform->dim_four;
    transform->size_transforms_arr = size_transforms_arr;

    //populate_strided_metadata(transform, seq_length);

    return transform;
}

transformations *generate_windowed_ds(int batch, int seq_length,
                                      int num_heads, int hidden_dimension, int sparsity_param)
{

    assert(seq_length >= sparsity_param && "Sequence length must be at least sparsity_parameter size!");

    // Window width is: 2*sparsity_param + 1.
    int dim_one = batch;
    int dim_two = num_heads;
    int dim_three = seq_length;
    int dim_four = std::min(2 * sparsity_param + 1, seq_length);
    int size_transforms_arr = seq_length;

    size_t tensor_size = dim_one * dim_two * dim_three * dim_four;

    // First allocate them on the heap.
    //   Set the necessary values within the transformations struct.
    transformations *transform = (transformations *)malloc(sizeof(transformations));
    transform->linear_transformation = (vector *)malloc(sizeof(vector) * dim_three);
    transform->linear_transformation_spmm = (vector_int *)malloc(sizeof(vector_int) * dim_three);
    transform->nnzs = (int *)malloc(sizeof(int) * dim_three);
    transform->dim_one = dim_one;
    transform->dim_two = dim_two;
    transform->dim_three = dim_three;
    transform->dim_four = dim_four;
    transform->total_value_size = transform->dim_three * transform->dim_four;
    transform->max_width = transform->dim_four;
    transform->size_transforms_arr = size_transforms_arr;

    for (int i = 0; i < dim_three; i++)
    {
        int dense_one, dense_two;

        if (i <= sparsity_param)
        { // Then
            dense_one = 0;
            dense_two = 1;
            // transform->nnzs[i] = std::min(sparsity_param + 1 + i, seq_length);
        }
        else if (i >= seq_length - sparsity_param)
        {
            dense_one = i - sparsity_param;
            dense_two = dense_one + 1;
        }
        else
        {
            dense_one = i - sparsity_param;
            dense_two = dense_one + 1;
        }

        int num_left = 0;
        int num_right = 0;

        if (i <= sparsity_param)
        {
            num_left += i;
        }
        else
        {
            num_left += sparsity_param;
        }

        if (i >= (seq_length - sparsity_param))
        {
            num_right += seq_length - i - 1;
        }
        else
        {
            num_right += sparsity_param;
        }

        transform->nnzs[i] = num_left + num_right + 1;

        Eigen::Matrix2f A;
        Eigen::Vector2f b;
        A << dense_one, 1, dense_two, 1;
        b << 0, 1;

        // Then we solve and retrieve the answer.
        Eigen::Vector2f x = A.colPivHouseholderQr().solve(b);

        // Finally, write to the linear transformations and vector
        //  arrays.
        vector current_transform = {x(0, 0), x(1, 0)};
        vector_int transform_spmm = {int(round(1 / x(0, 0))), int(round(-1 * x(1, 0)))};
        transform->linear_transformation[i] = current_transform;
        transform->linear_transformation_spmm[i] = transform_spmm;
    }
    return transform;
}

transformations *generate_blocked_ds(int batch, int seq_length,
                                     int num_heads, int hidden_dimension, int sparsity_param)
{
    // Over here, each row has width: 2*sparsity_param

    assert(seq_length % sparsity_param == 0 && "Sparsity parameter should divide the sequence length");
    int dim_one = batch;
    int dim_two = num_heads;
    int dim_three = seq_length;
    int dim_four = std::min(sparsity_param * 2, seq_length);
    int size_transforms_arr = seq_length;

    // We first allocate the data_structure values array.
    size_t tensor_size = dim_one * dim_two * dim_three * dim_four;

    // Next, we populate the linear_transformations.

    // First allocate them on the heap.
    transformations *transform = (transformations *)malloc(sizeof(transformations));
    transform->linear_transformation = (vector *)malloc(sizeof(vector) * dim_three);
    transform->linear_transformation_spmm = (vector_int *)malloc(sizeof(vector_int) * dim_three);
    transform->nnzs = (int *)malloc(sizeof(int) * dim_three);
    transform->dim_one = batch;
    transform->dim_two = num_heads;
    transform->dim_three = seq_length;
    transform->dim_four = std::min(2 * sparsity_param, seq_length);
    transform->total_value_size = transform->dim_three * transform->dim_four;
    transform->max_width = transform->dim_four;
    transform->size_transforms_arr = size_transforms_arr;

    // Next, we populate them with values.
    // These transformations are from: dense -> sparse coordinate mappings.
    for (int i = 0; i < dim_three; i++)
    {
        // First we figure out the block number;
        int block_num = i / sparsity_param;
        // Then we compute the real dense coordinate of the
        // first two points in the row.

        int dense_one, dense_two;

        if (block_num < 1)
        {
            dense_one = 0;
            dense_two = 1;
            transform->nnzs[i] = std::min(sparsity_param, seq_length);
        }
        else
        {
            dense_one = (block_num - 1) * sparsity_param;
            dense_two = dense_one + 1;
            transform->nnzs[i] = std::min(2 * sparsity_param, seq_length);
        }

        Eigen::Matrix2f A;
        Eigen::Vector2f b;
        A << dense_one, 1, dense_two, 1;
        b << 0, 1;

        // Then we solve and retrieve the answer.
        Eigen::Vector2f x = A.colPivHouseholderQr().solve(b);

        // Finally, write to the linear transformations and vector
        //  arrays.
        vector current_transform = {x(0, 0), x(1, 0)};
        vector_int transform_spmm = {int(round(1 / x(0, 0))), int(round(-1 * x(1, 0)))};
        transform->linear_transformation[i] = current_transform;
        transform->linear_transformation_spmm[i] = transform_spmm;
    }

    return transform;
}

void pretty_print(transformations *transforms)
{
    // We pretty print all the information, for ease of debugging.
    printf("Dim one: %d, Dim two: %d, Dim three: %d, Dim four: %d\n",
           transforms->dim_one, transforms->dim_two, transforms->dim_three, transforms->dim_four);

    printf("Printing transformations:\n");
    for (int i = 0; i < transforms->size_transforms_arr; i++)
    {
        printf("Dense to Sparse Row %d: a : %0.3f, b: %0.3f, nnzs: %d, Sparse to Dense: a : %d, b: %d\n", i + 1, transforms->linear_transformation[i].x, transforms->linear_transformation[i].y, transforms->nnzs[i], transforms->linear_transformation_spmm[i].x, transforms->linear_transformation_spmm[i].y);
    }
}
