#ifndef SRC_LAYER_CPU_NEW_FORWARD_H
#define SRC_LAYER_CPU_NEW_FORWARD_H

void conv_forward_cpu(float *  output, const float *  input, const float *  mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S = 1);
void unfold(float* device_input_ptr, float *device_input_rearrange, int C , int K , int H ,int W ,int S, int B);

#endif // SRC_LAYER_CPU_NEW_FORWARD_H
