"""
Patch for DFlash + Qwen3.5 Mamba-hybrid KV cache page size mismatch.
Replaces LCM-based alignment with GCD-based alignment in unify_kv_cache_spec_page_size.
Ref: https://github.com/z-lab/dflash/issues/139
     https://forums.developer.nvidia.com/t/dflash-for-qwen3-5/374328?page=2
"""
import importlib
import math

TARGET_FILE = "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py"

with open(TARGET_FILE, "r") as f:
    source = f.read()

# The buggy assertion on line ~1030:
#   assert new_spec.page_size_bytes == max_page_size
# This fails because LCM scaling creates a page size the drafter can't divide evenly.
# Fix: replace the strict assert with GCD-based re-alignment.

old_code = "assert new_spec.page_size_bytes == max_page_size"
new_code = """if new_spec.page_size_bytes != max_page_size:
                # GCD-based fallback for hybrid architectures (Mamba + Attention + DFlash)
                import math as _math
                _gcd = _math.gcd(new_spec.page_size_bytes, max_page_size)
                if max_page_size % _gcd == 0:
                    max_page_size = new_spec.page_size_bytes
                else:
                    raise AssertionError(
                        f"Cannot align page sizes: {new_spec.page_size_bytes} vs {max_page_size}")"""

if old_code in source:
    source = source.replace(old_code, new_code)
    with open(TARGET_FILE, "w") as f:
        f.write(source)
    print("[PATCH] Successfully patched kv_cache_utils.py (GCD-based alignment)")
else:
    print("[PATCH] Target assertion not found - may already be patched or different vLLM version")
