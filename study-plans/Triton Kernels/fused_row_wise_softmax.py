import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, out_ptr, x_row_stride, out_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    x_row = x_ptr + row * x_row_stride
    x_vals = tl.load(x_row + offsets, mask=mask, other=-float('inf'))
    row_max = tl.max(x_vals, axis=0)
    x_shifted = x_vals - row_max
    x_exp = tl.exp(x_shifted)
    row_sum = tl.sum(x_exp, axis=0)
    out = x_exp / row_sum
    out_row = out_ptr + row * out_row_stride
    tl.store(out_row + offsets, out, mask=mask)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch softmax_kernel with one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    softmax_kernel[grid](
        x, out, x.stride(0), out.stride(0), N, BLOCK_SIZE=BLOCK_SIZE,
    )