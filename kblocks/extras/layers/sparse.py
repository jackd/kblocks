from wtftf.meta import layered

from kblocks.ops import sparse as _sparse_ops

block_diagonalize_sparse = layered(_sparse_ops.block_diagonalize_sparse)
apply_offset = layered(_sparse_ops.apply_offset)
block_diagonalize_sparse_general = layered(_sparse_ops.block_diagonalize_sparse_general)
ragged_to_sparse_indices = layered(_sparse_ops.ragged_to_sparse_indices)
unstack = layered(_sparse_ops.unstack)
remove_dim = layered(_sparse_ops.remove_dim)
remove_leading_dim = layered(_sparse_ops.remove_leading_dim)
