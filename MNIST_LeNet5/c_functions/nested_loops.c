#include <stdio.h>

void count_conv_multiplications (int input_size_0, int input_size_1, int input_size_2, int input_size_3,
				int weight_size_0, int weight_size_2, int weight_size_3,
				int stride_0, int stride_1, double *input,
				int *num_all_mul, int *num_non_zero_mul) {

    // Initialize the counters
    *num_all_mul = 0;
    *num_non_zero_mul = 0;

    // Iterate over the nested loops
    for (int b = 0; b < input_size_0; b++) { // Batch dimension
        for (int c_out = 0; c_out < weight_size_0; c_out++) { // Output channel dimension
            for (int h_out = 0; h_out <= input_size_2 - weight_size_2; h_out += stride_0) { // Output Height dimension
                for (int w_out = 0; w_out <= input_size_3 - weight_size_3; w_out += stride_1) { // Output Width dimension
                    for (int c_in = 0; c_in < input_size_1; c_in++) { // Input/Kernel channel dimension
                        for (int h_k = 0; h_k < weight_size_2; h_k++) { // Kernel height dimension
                            for (int w_k = 0; w_k < weight_size_3; w_k++) { // Kernel width dimension

                                // Calculate the input index
                                int input_index = ((b * input_size_1 + c_in) * input_size_2 + (h_out + h_k)) * input_size_3 + (w_out + w_k);
                                
                                // Get the input operand
                                double input_operand = input[input_index];
                                
                                // Increment the all multiplications counter
                                (*num_all_mul)++;
                                
                                // Check for non-zero input and increment the non-zero counter
                                if (input_operand != 0) {
					(*num_non_zero_mul)++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

}

void count_fc_multiplications(int input_size_0, int input_size_1, int weight_size_0,
                              double *input, int *num_all_mul, int *num_non_zero_mul) {

    // Initialize the counters
    *num_all_mul = 0;
    *num_non_zero_mul = 0;

    for (int i = 0; i < input_size_0; i++) { // Iterate over batch dimension
        for (int j = 0; j < weight_size_0; j++) { // Iterate over output dimension
            for (int k = 0; k < input_size_1; k++) { // Iterate over the identical-size dimension
                (*num_all_mul)++;
                double input_value = input[i * input_size_1 + k]; // Calculate the index for the 2D input array
                if (input_value != 0) {
                    (*num_non_zero_mul)++;
                }
            }
        }
    }
}
