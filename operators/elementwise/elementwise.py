import argparse
from typing import Callable

import torch
import triton
import triton.language as tl


# ============================================================
# Triton kernels
# ============================================================


@triton.jit
def elementwise_v0_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """V0：基础连续分块版本。每个 Triton program 处理一个连续数据块。"""
    program_id = tl.program_id(axis=0)
    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)


@triton.jit
def elementwise_v1_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """V1：grid-stride 版本。每个 program 跨步处理多个连续数据块。"""
    program_id = tl.program_id(axis=0)
    block_start = program_id * BLOCK_SIZE
    grid_stride = tl.num_programs(axis=0) * BLOCK_SIZE

    while block_start < num_elements:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements

        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        tl.store(c_ptr + offsets, a + b, mask=mask)

        block_start += grid_stride


@triton.jit
def elementwise_v2_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
    VEC_SIZE: tl.constexpr,
):
    """V2：显式多元素版本。每个逻辑位置一次处理 VEC_SIZE 个连续元素。"""
    program_id = tl.program_id(axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    vec_offsets = tl.arange(0, VEC_SIZE)

    base = program_id * BLOCK_SIZE * VEC_SIZE
    offsets = base + lanes[:, None] * VEC_SIZE + vec_offsets[None, :]
    mask = offsets < num_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)


@triton.jit
def elementwise_v3_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
    UNROLL: tl.constexpr,
):
    """V3：静态展开版本。每个 program 连续处理 UNROLL 个数据块。"""
    program_id = tl.program_id(axis=0)
    program_base = program_id * BLOCK_SIZE * UNROLL
    lane_offsets = tl.arange(0, BLOCK_SIZE)

    for i in tl.static_range(0, UNROLL):
        offsets = program_base + i * BLOCK_SIZE + lane_offsets
        mask = offsets < num_elements

        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        tl.store(c_ptr + offsets, a + b, mask=mask)


def _check_inputs(a: torch.Tensor, b: torch.Tensor) -> None:
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("a 和 b 必须是 CUDA 张量。")
    if a.shape != b.shape:
        raise ValueError("a 和 b 的形状必须相同。")
    if a.dtype != b.dtype:
        raise ValueError("a 和 b 的数据类型必须相同。")
    if a.dtype != torch.float32:
        raise ValueError("当前示例只测试 torch.float32。")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("a 和 b 必须是连续张量。")


def elementwise_v0(
    a: torch.Tensor,
    b: torch.Tensor,
    block_size: int = 256,
) -> torch.Tensor:
    _check_inputs(a, b)
    c = torch.empty_like(a)
    num_elements = a.numel()
    grid = (triton.cdiv(num_elements, block_size),)
    elementwise_v0_kernel[grid](a, b, c, num_elements, BLOCK_SIZE=block_size)
    return c


def elementwise_v1(
    a: torch.Tensor,
    b: torch.Tensor,
    block_size: int = 256,
    programs_per_sm: int = 4,
) -> torch.Tensor:
    _check_inputs(a, b)
    c = torch.empty_like(a)
    num_elements = a.numel()

    num_tiles = triton.cdiv(num_elements, block_size)
    sm_count = torch.cuda.get_device_properties(a.device).multi_processor_count
    num_programs = max(1, min(num_tiles, sm_count * programs_per_sm))

    elementwise_v1_kernel[(num_programs,)](
        a,
        b,
        c,
        num_elements,
        BLOCK_SIZE=block_size,
    )
    return c


def elementwise_v2(
    a: torch.Tensor,
    b: torch.Tensor,
    block_size: int = 256,
    vec_size: int = 4,
) -> torch.Tensor:
    _check_inputs(a, b)
    c = torch.empty_like(a)
    num_elements = a.numel()
    elements_per_program = block_size * vec_size
    grid = (triton.cdiv(num_elements, elements_per_program),)

    elementwise_v2_kernel[grid](
        a,
        b,
        c,
        num_elements,
        BLOCK_SIZE=block_size,
        VEC_SIZE=vec_size,
    )
    return c


def elementwise_v3(
    a: torch.Tensor,
    b: torch.Tensor,
    block_size: int = 256,
    unroll: int = 4,
) -> torch.Tensor:
    _check_inputs(a, b)
    c = torch.empty_like(a)
    num_elements = a.numel()
    elements_per_program = block_size * unroll
    grid = (triton.cdiv(num_elements, elements_per_program),)

    elementwise_v3_kernel[grid](
        a,
        b,
        c,
        num_elements,
        BLOCK_SIZE=block_size,
        UNROLL=unroll,
    )
    return c


# Correctness and timing
def benchmark_ms(
    function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    a: torch.Tensor,
    b: torch.Tensor,
    warmup: int = 20,
    repeats: int = 100,
) -> float:
    for _ in range(warmup):
        function(a, b)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeats):
        function(a, b)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / repeats


def check_and_benchmark(num_elements: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境无法使用 CUDA。")

    print("Python/PyTorch 环境检查")
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA:", torch.version.cuda)
    print("Triton:", triton.__version__)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
    print("元素数量:", num_elements)
    print()

    torch.manual_seed(42)
    a = torch.rand(num_elements, device="cuda", dtype=torch.float32)
    b = torch.rand(num_elements, device="cuda", dtype=torch.float32)
    reference = a + b

    implementations: list[tuple[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]] = [
        ("PyTorch", lambda x, y: x + y),
        ("Triton V0 基础版", elementwise_v0),
        ("Triton V1 grid-stride", elementwise_v1),
        ("Triton V2 vec4", elementwise_v2),
        ("Triton V3 unroll4", elementwise_v3),
    ]

    for name, function in implementations:
        result = function(a, b)
        max_error = (result - reference).abs().max().item()
        correct = torch.allclose(result, reference, rtol=1e-5, atol=1e-6)
        elapsed = benchmark_ms(function, a, b)
        print(f"{name:<24} 正确={str(correct):<5} 最大误差={max_error:.8f} 平均耗时={elapsed:.6f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton elementwise add V0-V3")
    parser.add_argument(
        "--num-elements",
        type=int,
        default=10_000_000,
        help="向量元素数量，默认 10,000,000。",
    )
    args = parser.parse_args()
    check_and_benchmark(args.num_elements)
