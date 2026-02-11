import torch
from torch import nn
import triton
import triton.language as tl
import math

# Try importing flash_attn, fallback to xformers or sdpa if not available
try:
    # Check capability to avoid using flash_attn on unsupported hardware (e.g. T4) even if installed
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if major < 8:
            raise ImportError("FlashAttention requires Ampere or newer")

    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    try:
        import xformers.ops as xops
        XFORMERS_AVAILABLE = True
    except ImportError:
        XFORMERS_AVAILABLE = False

from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


# Fallback implementations
if not FLASH_ATTN_AVAILABLE:
    def flash_attn_varlen_func(q, k, v, max_seqlen_q, cu_seqlens_q, max_seqlen_k, cu_seqlens_k, softmax_scale=None, causal=True, block_table=None):
        # Fallback for prefill phase
        # q: [total_tokens, num_heads, head_dim]
        # k, v: [total_tokens, num_heads, head_dim] OR [num_blocks, block_size, num_kv_heads, head_dim] if block_table is present

        batch_size = len(cu_seqlens_q) - 1
        outputs = []

        for i in range(batch_size):
            # Extract query for this sequence
            start_q, end_q = cu_seqlens_q[i], cu_seqlens_q[i+1]
            qi = q[start_q:end_q].unsqueeze(0)  # [1, seq_len_q, num_heads, head_dim]

            # Extract key/value
            if block_table is not None:
                # Paged attention for prefill (e.g. prefix cache)
                # k, v are actually k_cache, v_cache
                start_k, end_k = cu_seqlens_k[i], cu_seqlens_k[i+1]
                seq_len_k = end_k - start_k

                block_size = k.shape[1]
                num_blocks = (seq_len_k + block_size - 1) // block_size
                block_indices = block_table[i, :num_blocks]

                # Gather blocks: [num_blocks, block_size, num_kv_heads, head_dim]
                k_blocks = k[block_indices]
                v_blocks = v[block_indices]

                # Flatten
                ki = k_blocks.view(-1, k_blocks.shape[-2], k_blocks.shape[-1])
                vi = v_blocks.view(-1, v_blocks.shape[-2], v_blocks.shape[-1])

                # Trim to actual sequence length
                ki = ki[:seq_len_k].unsqueeze(0)
                vi = vi[:seq_len_k].unsqueeze(0)
            else:
                # Standard attention
                start_k, end_k = cu_seqlens_k[i], cu_seqlens_k[i+1]
                ki = k[start_k:end_k].unsqueeze(0)
                vi = v[start_k:end_k].unsqueeze(0)

            # Perform attention
            if XFORMERS_AVAILABLE:
                # Create attention bias for causal masking if needed
                attn_bias = None
                if causal:
                    # Lower triangular mask
                    attn_bias = xops.LowerTriangularMask()

                out_i = xops.memory_efficient_attention(
                    qi, ki, vi, attn_bias=attn_bias, scale=softmax_scale
                )
            else:
                # SDPA
                qi_t = qi.transpose(1, 2)
                ki_t = ki.transpose(1, 2)
                vi_t = vi.transpose(1, 2)

                # Handle GQA
                if qi_t.shape[1] != ki_t.shape[1]:
                    rep = qi_t.shape[1] // ki_t.shape[1]
                    ki_t = ki_t.repeat_interleave(rep, dim=1)
                    vi_t = vi_t.repeat_interleave(rep, dim=1)

                if softmax_scale is None:
                    softmax_scale = 1.0 / math.sqrt(qi_t.shape[-1])

                out_i = torch.nn.functional.scaled_dot_product_attention(
                    qi_t, ki_t, vi_t, is_causal=causal, scale=softmax_scale
                )
                out_i = out_i.transpose(1, 2)

            outputs.append(out_i.squeeze(0))

        return torch.cat(outputs, dim=0)

    def flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens, block_table, softmax_scale=None, causal=True):
        # Fallback for decode phase (PagedAttention)

        batch_size = q.shape[0]
        block_size = k_cache.shape[1]
        outputs = []

        for i in range(batch_size):
            # Gather K and V for this sequence
            seq_len = cache_seqlens[i].item()
            num_blocks = (seq_len + block_size - 1) // block_size

            block_indices = block_table[i, :num_blocks]

            # Gather blocks
            k_blocks = k_cache[block_indices]
            v_blocks = v_cache[block_indices]

            # Flatten
            k_seq = k_blocks.view(-1, k_blocks.shape[-2], k_blocks.shape[-1])
            v_seq = v_blocks.view(-1, v_blocks.shape[-2], v_blocks.shape[-1])

            # Trim
            k_seq = k_seq[:seq_len]
            v_seq = v_seq[:seq_len]

            # Add batch dim
            k_seq = k_seq.unsqueeze(0)
            v_seq = v_seq.unsqueeze(0)

            qi = q[i:i+1] # [1, 1, num_heads, head_dim]

            if XFORMERS_AVAILABLE:
                out_i = xops.memory_efficient_attention(
                    qi, k_seq, v_seq, scale=softmax_scale
                )
            else:
                qi_t = qi.transpose(1, 2)
                ki_t = k_seq.transpose(1, 2)
                vi_t = v_seq.transpose(1, 2)

                if qi_t.shape[1] != ki_t.shape[1]:
                    rep = qi_t.shape[1] // ki_t.shape[1]
                    ki_t = ki_t.repeat_interleave(rep, dim=1)
                    vi_t = vi_t.repeat_interleave(rep, dim=1)

                if softmax_scale is None:
                    softmax_scale = 1.0 / math.sqrt(qi_t.shape[-1])

                out_i = torch.nn.functional.scaled_dot_product_attention(qi_t, ki_t, vi_t, is_causal=False, scale=softmax_scale)
                out_i = out_i.transpose(1, 2)

            outputs.append(out_i)

        return torch.cat(outputs, dim=0)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
