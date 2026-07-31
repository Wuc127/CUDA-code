#include "reduce_cpu.h"


float reduce_cpu(const float* input, int size)
{
    float sum = 0.0f;

    for (int i = 0; i < size; i++)
    {
        sum += input[i];
    }

    return sum;
}