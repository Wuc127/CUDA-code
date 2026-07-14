#pragma once

void attention_cpu(
    const float* Q,   // Q: [bs, len_q, dim]
    const float* K,   // K: [bs, len_kv, dim]
    const float* V,   // V: [bs, len_kv, dim]
    float* S,         // S: [bs, len_q, len_kv]
    float* P,         // P: [bs, len_q, len_kv]
    float* O,         // O: [bs, len_q, dim]
    int bs,
    int len_q,
    int len_kv,
    int dim
);