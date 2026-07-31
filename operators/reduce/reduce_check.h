#pragma once

bool reduce_check(const float* cpu_result,
                  const float* gpu_result,
                  float eps = 1e-4f);