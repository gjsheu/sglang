import torch

from sglang.jit_kernel.diffusion.triton.scale_shift import fuse_scale_shift_kernel
from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp


class MulAdd(CustomOp):
    """
    Fuse elementwise mul and add
    Input: a, b, c, OptionalInt[k]
    Output: a * (k + b) + c
    """

    def __init__(self, prefix: str = ""):
        super().__init__()

    def forward_native(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, k: int = 0
    ) -> torch.Tensor:
        # a.shape: [batch_size, seq_len, inner_dim]
        if b.dim() == 4:
            # b.shape: [batch_size, num_frames, 1, inner_dim]
            num_frames = b.shape[1]
            frame_seqlen = a.shape[1] // num_frames
            return c + (
                a.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (k + b)
            ).flatten(1, 2)
        else:
            # b.shape: [batch_size, 1, inner_dim]
            return c + a * (k + b)

    def forward_cuda(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, k: int = 0
    ):
        return fuse_scale_shift_kernel(a, b, c, scale_constant=k)

    def forward_npu(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, k: int = 0
    ):
        block_l, block_c = 128, 64
        B, L, C = a.shape[0], a.shape[1], a.shape[2]
        if B * L * C / block_l / block_c < 65535:
            from sgl_kernel_npu.norm.scale_shift import fused_scale_shift

            return fused_scale_shift(
                a, b, c, scale_constant=k, block_l=block_l, block_c=block_c
            )
        else:
            return self.forward_native(a, b, c, k)
