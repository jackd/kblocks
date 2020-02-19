# pylint: disable=import-error,no-name-in-module
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import segment_id_ops
# pylint: enable=import-error,no-name-in-module


class RaggedStructure(object):

  def __init__(self,
               row_splits,
               cached_row_lengths=None,
               cached_value_rowids=None,
               cached_nrows=None,
               internal=False):
    """Creates a `RaggedStructure` with a specified partitioning for `values`.

    This constructor is private -- please use one of the following ops to
    build `RaggedStructure`s:

      * `tf.RaggedStructure.from_row_lengths`
      * `tf.RaggedStructure.from_value_rowids`
      * `tf.RaggedStructure.from_row_splits`
      * `tf.RaggedStructure.from_row_starts`
      * `tf.RaggedStructure.from_row_limits`

    Args:
      values: A potentially ragged tensor of any dtype and shape `[nvals, ...]`.
      row_splits: A 1-D int64 tensor with shape `[nrows+1]`.
      cached_row_lengths: A 1-D int64 tensor with shape `[nrows]`
      cached_value_rowids: A 1-D int64 tensor with shape `[nvals]`.
      cached_nrows: A 1-D int64 scalar tensor.
      internal: True if the constructor is being called by one of the factory
        methods.  If false, an exception will be raised.

    Raises:
      TypeError: If a row partitioning tensor has an inappropriate dtype.
      TypeError: If exactly one row partitioning argument was not specified.
      ValueError: If a row partitioning tensor has an inappropriate shape.
      ValueError: If multiple partitioning arguments are specified.
      ValueError: If nrows is specified but value_rowids is not None.
    """
    if not internal:
      raise ValueError("RaggedTensor constructor is private; please use one "
                       "of the factory methods instead (e.g., "
                       "RaggedTensor.from_row_lengths())")

    # Validate the arguments.
    if not isinstance(row_splits, ops.Tensor):
      raise TypeError("Row-partitioning argument must be a Tensor.")
    row_splits.shape.assert_has_rank(1)
    row_splits.set_shape([None])

    self._row_splits = row_splits

    # Store any cached tensors.  These are used to avoid unnecessary
    # round-trip conversions when a RaggedTensor is constructed from
    # lengths or rowids, and we later want those lengths/rowids back.
    for tensor in [cached_row_lengths, cached_value_rowids, cached_nrows]:
      if tensor is not None and not isinstance(tensor, ops.Tensor):
        raise TypeError("Cached value must be a Tensor or None.")
    self._cached_row_lengths = cached_row_lengths
    self._cached_value_rowids = cached_value_rowids
    self._cached_nrows = cached_nrows

  #=============================================================================
  # Factory Methods
  #=============================================================================

  @classmethod
  def from_value_rowids(cls, value_rowids, nrows=None, name=None):
    """Creates a `RaggedStructure` with rows partitioned by `value_rowids`.

    The returned `RaggedStructure` corresponds with provied value_rowids. The
    corresponding `RaggedTensor` would be given by

    ```python
    result = [[values[i] for i in range(len(values)) if value_rowids[i] == row]
              for row in range(nrows)]
    ```

    Warning: currently, this needs to cast value_rowids to int64 before
    converting, since `tf.bincount` only supports `int32`.

    Args:
      value_rowids: A 1-D int64 tensor with shape `[nvals]`, which corresponds
        one-to-one with `values`, and specifies each value's row index.  Must be
        nonnegative, and must be sorted in ascending order.
      nrows: An int64 scalar specifying the number of rows.  This should be
        specified if the `RaggedTensor` may containing empty training rows. Must
        be greater than `value_rowids[-1]` (or zero if `value_rowids` is empty).
        Defaults to `value_rowids[-1]` (or zero if `value_rowids` is empty).
      name: A name prefix for the RaggedStructure (optional).

    Returns:
      A `RaggedStructure`.  `result.rank = values.rank + 1`.
      `result.ragged_rank = values.ragged_rank + 1`.

    Raises:
      ValueError: If `nrows` is incompatible with `value_rowids`.

    #### Example:
      ```python
      >>> print(tf.RaggedStructure.from_value_rowids(
      ...     values=[3, 1, 4, 1, 5, 9, 2, 6],
      ...     value_rowids=[0, 0, 0, 0, 2, 2, 2, 3],
      ...     nrows=5))
      <tf.RaggedStructure [0, 4, 7, 10]>
      ```
    """
    with ops.name_scope(name, "RaggedFromValueRowIds",
                        [value_rowids, nrows]):
      value_rowids = ops.convert_to_tensor(
          value_rowids, dtypes.int64, name="value_rowids")
      if nrows is None:
        const_rowids = tensor_util.constant_value(value_rowids)
        if const_rowids is None:
          nrows = array_ops.concat([value_rowids[-1:], [-1]], axis=0)[0] + 1
          const_nrows = None
        else:
          const_nrows = const_rowids[-1] + 1 if const_rowids.size > 0 else 0
          nrows = ops.convert_to_tensor(const_nrows, dtypes.int64, name="nrows")
      else:
        nrows = ops.convert_to_tensor(nrows, dtypes.int64, "nrows")
        const_nrows = tensor_util.constant_value(nrows)
        if const_nrows is not None:
          if const_nrows < 0:
            raise ValueError("Expected nrows >= 0; got %d" % const_nrows)
          const_rowids = tensor_util.constant_value(value_rowids)
          if const_rowids is not None and const_rowids.size > 0:
            if not const_nrows >= const_rowids[-1] + 1:
              raise ValueError(
                  "Expected nrows >= value_rowids[-1] + 1; got nrows=%d, "
                  "value_rowids[-1]=%d" % (const_nrows, const_rowids[-1]))

      value_rowids.shape.assert_has_rank(1)
      nrows.shape.assert_has_rank(0)

      # Convert value_rowids & nrows to row_splits.
      # Note: we don't use segment_ids_to_row_splits() here because we want
      # to save the intermediate value `row_lengths`, so we can cache it.
      # TODO(b/116708836) Upgrade bincount to accept int64 so we can skip the
      # cast (Remove the warning in the docstring when we do.)
      value_rowids_int32 = math_ops.cast(value_rowids, dtypes.int32)
      nrows_int32 = math_ops.cast(nrows, dtypes.int32)
      row_lengths = math_ops.bincount(
          value_rowids_int32,
          minlength=nrows_int32,
          maxlength=nrows_int32,
          dtype=dtypes.int64)
      row_splits = array_ops.concat([[0], math_ops.cumsum(row_lengths)], axis=0)
      if const_nrows is not None:
        row_lengths.set_shape([const_nrows])
        row_splits.set_shape([const_nrows + 1])

      return cls(
          row_splits,
          cached_row_lengths=row_lengths,
          cached_value_rowids=value_rowids,
          cached_nrows=nrows,
          internal=True)

  @classmethod
  def from_row_splits(cls, row_splits, name=None):
    """Creates a `RaggedStructure` with rows partitioned by `row_splits`.

    The returned `RaggedStructure` corresponds with the python list defined by:

    ```python
    result = [values[row_splits[i]:row_splits[i + 1]]
              for i in range(len(row_splits) - 1)]
    ```

    Args:
      row_splits: A 1-D int64 tensor with shape `[nrows+1]`.  Must not be empty,
        and must be sorted in ascending order.  `row_splits[0]` must be zero and
        `row_splits[-1]` must be `nvals`.
      name: A name prefix for the RaggedStructure (optional).

    Returns:
      A `RaggedStructure`.

    Raises:
      ValueError: If `row_splits` is an empty list.

    #### Example:
      ```python
      >>> print(tf.RaggedTensor.from_row_splits(
      ...     row_splits=[0, 4, 4, 7, 8, 8]))
      <tf.RaggedTensor [0, 4, 4, 7, 8, 8]>
      ```
    """
    if isinstance(row_splits, (list, tuple)) and not row_splits:
      raise ValueError("row_splits tensor may not be empty.")
    with ops.name_scope(name, "RaggedFromRowSplits", [row_splits]):
      row_splits = ops.convert_to_tensor(row_splits, dtypes.int64, "row_splits")
      row_splits.shape.assert_has_rank(1)
      return cls(row_splits=row_splits, internal=True)

  @classmethod
  def from_row_lengths(cls, row_lengths, name=None):
    """Creates a `RaggedStructure` with rows partitioned by `row_lengths`.

    The returned `RaggedStructure` corresponds with the python list defined by:

    ```python
    result = [[values.pop(0) for i in range(length)]
              for length in row_lengths]
    ```

    Args:
      row_lengths: A 1-D int64 tensor with shape `[nrows]`.  Must be
        nonnegative.  `sum(row_lengths)` must be `nvals`.
      name: A name prefix for the RaggedStructure (optional).

    Returns:
      A `RaggedStructure`.

    #### Example:
      ```python
      >>> print(tf.RaggedTensor.from_row_lengths(
      ...     row_lengths=[4, 0, 3, 1, 0]))
      <tf.RaggedStructure [0, 4, 7, 8, 8])>
      ```
    """
    with ops.name_scope(name, "RaggedFromRowLengths", [row_lengths]):
      row_lengths = ops.convert_to_tensor(row_lengths, dtypes.int64,
                                          "row_lengths")
      row_lengths.shape.assert_has_rank(1)
      row_limits = math_ops.cumsum(row_lengths)
      row_splits = array_ops.concat([[0], row_limits], axis=0)
      return cls(
          row_splits=row_splits,
          cached_row_lengths=row_lengths,
          internal=True)

  @classmethod
  def from_row_starts(cls, row_starts, nvals, name=None):
    """Creates a `RaggedStructure` with rows partitioned by `row_starts`.

    Equivalent to: `from_row_splits(concat([row_starts, nvals]))`.

    Args:
      row_starts: A 1-D int64 tensor with shape `[nrows]`.  Must be nonnegative
        and sorted in ascending order.  If `nrows>0`, then `row_starts[0]` must
        be zero.
      nvals: An int64 scalar. Must be >= row_starts[-1]
      name: A name prefix for the RaggedStructure (optional).

    Returns:
      A `RaggedStructure`.  `result.rank = values.rank + 1`.
      `result.ragged_rank = values.ragged_rank + 1`.

    #### Example:
      ```python
      >>> print(tf.RaggedTensor.from_row_starts(
      ...     values=[3, 1, 4, 1, 5, 9, 2, 6],
      ...     row_starts=[0, 4, 4, 7, 8]))
      <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>
      ```
    """
    with ops.name_scope(name, "RaggedFromRowStarts", [row_starts]):
      row_starts = ops.convert_to_tensor(row_starts, dtypes.int64, "row_starts")
      row_starts.shape.assert_has_rank(1)
      row_splits = array_ops.concat([row_starts, nvals], axis=0)
      return cls(row_splits=row_splits, internal=True)

  @classmethod
  def from_row_limits(cls,  row_limits, name=None):
    """Creates a `RaggedTensor` with rows partitioned by `row_limits`.

    Equivalent to: `from_row_splits(values, concat([0, row_limits]))`.

    Args:
      row_limits: A 1-D int64 tensor with shape `[nrows]`.  Must be sorted in
        ascending order.  If `nrows>0`, then `row_limits[-1]` must be `nvals`.
      name: A name prefix for the RaggedTensor (optional).

    Returns:
      A `RaggedTensor`.  `result.rank = values.rank + 1`.
      `result.ragged_rank = values.ragged_rank + 1`.

    #### Example:
      ```python
      >>> print(tf.RaggedTensor.from_row_limits(
      ...     values=[3, 1, 4, 1, 5, 9, 2, 6],
      ...     row_limits=[4, 4, 7, 8, 8]))
      <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>
      ```
    """
    with ops.name_scope(name, "RaggedFromRowLimits", [row_limits]):
      row_limits = ops.convert_to_tensor(row_limits, dtypes.int64, "row_limits")
      row_limits.shape.assert_has_rank(1)
      zero = array_ops.zeros([1], dtypes.int64)
      row_splits = array_ops.concat([zero, row_limits], axis=0)
      return cls(row_splits=row_splits, internal=True)

  @property
  def row_splits(self):
    """The row-split indices for this ragged tensor's `values`.

    `rt.row_splits` specifies where the values for each row begin and end in
    `rt.values`.  In particular, the values for row `rt[i]` are stored in
    the slice `rt.values[rt.row_splits[i]:rt.row_splits[i+1]]`.

    Returns:
      A 1-D `int64` `Tensor` with shape `[self.nrows+1]`.
      The returned tensor is non-empty, and is sorted in ascending order.
      `self.row_splits[0]` is zero, and `self.row_splits[-1]` is equal to
      `self.values.shape[0]`.

    #### Example:
      ```python
      >>> rt = ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
      >>> print rt.row_splits  # indices of row splits in rt.values
      tf.Tensor([0, 4, 4, 7, 8, 8])
      ```
    """
    return self._row_splits

  def value_rowids(self, name=None):
    """Returns the row indices for the `values` in this ragged tensor.

    `rt.value_rowids()` corresponds one-to-one with the outermost dimension of
    `rt.values`, and specifies the row containing each value.  In particular,
    the row `rt[row]` consists of the values `rt.values[j]` where
    `rt.value_rowids()[j] == row`.

    Args:
      name: A name prefix for the returned tensor (optional).

    Returns:
      A 1-D `int64` `Tensor` with shape `self.values.shape[:1]`.
      The returned tensor is nonnegative, and is sorted in ascending order.

    #### Example:
      ```python
      >>> rt = ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
      >>> rt.values
      tf.Tensor([3, 1, 4, 1, 5, 9, 2, 6])
      >>> rt.value_rowids()
      tf.Tensor([0, 0, 0, 0, 2, 2, 2, 3])  # corresponds 1:1 with rt.values
      ```
    """
    if self._cached_value_rowids is None:
      with ops.name_scope(name, "RaggedValueRowIds", [self]):
        self._cached_value_rowids = segment_id_ops.row_splits_to_segment_ids(self.row_splits)
    return self._cached_value_rowids



  def nrows(self, out_type=dtypes.int64, name=None):
    """Returns the number of rows in this ragged tensor.

    I.e., the size of the outermost dimension of the tensor.

    Args:
      out_type: `dtype` for the returned tensor.
      name: A name prefix for the returned tensor (optional).

    Returns:
      A scalar `Tensor` with dtype `out_type`.

    #### Example:
      ```python
      >>> rt = ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
      >>> rt.nrows()  # rt has 5 rows.
      5
      ```
    """
    if self._cached_nrows is None:
      with ops.name_scope(name, "RaggedNRows", [self]):
        self._cached_nrows = array_ops.shape(
            self.row_splits, out_type=out_type)[0] - 1
    return self._cached_nrows

  def row_starts(self, name=None):
    """Returns the start indices for rows in this ragged tensor.

    These indices specify where the values for each row begin in
    `self.values`.  `rt.row_starts()` is equal to `rt.row_splits[:-1]`.

    Args:
      name: A name prefix for the returned tensor (optional).

    Returns:
      A 1-D Tensor of int64 with shape `[nrows]`.
      The returned tensor is nonnegative, and is sorted in ascending order.

    #### Example:
      ```python
      >>> rt = ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
      >>> rt.values
      tf.Tensor([3, 1, 4, 1, 5, 9, 2, 6])
      >>> rt.row_starts()  # indices of row starts in rt.values
      tf.Tensor([0, 4, 4, 7, 8])
      ```
    """
    with ops.name_scope(name, "RaggedRowStarts", [self]):
      return self.row_splits[:-1]

  def row_limits(self, name=None):
    """Returns the limit indices for rows in this ragged tensor.

    These indices specify where the values for each row end in
    `self.values`.  `rt.row_limits(self)` is equal to `rt.row_splits[:-1]`.

    Args:
      name: A name prefix for the returned tensor (optional).

    Returns:
      A 1-D Tensor of int64 with shape `[nrows]`.
      The returned tensor is nonnegative, and is sorted in ascending order.

    #### Example:
      ```python
      >>> rt = ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
      >>> rt.values
      tf.Tensor([3, 1, 4, 1, 5, 9, 2, 6])
      >>> rt.row_limits()  # indices of row limits in rt.values
      tf.Tensor([4, 4, 7, 8, 8])
      ```
    """
    with ops.name_scope(name, "RaggedRowLimits", [self]):
      return self.row_splits[1:]

  def row_lengths(self, name=None):
    """Returns the lengths of the rows in this ragged tensor.

    `rt.row_lengths()[i]` indicates the number of values in the
    `i`th row of `rt`.

    Args:
      name: A name prefix for the returned tensor (optional).

    Returns:
      A potentially ragged Tensor of int64 with shape `self.shape[:axis]`.

    Raises:
      ValueError: If `axis` is out of bounds.

    #### Example:
      ```python
      >>> rt = ragged.constant([[[3, 1, 4], [1]], [], [[5, 9], [2]], [[6]], []])
      >>> rt.row_lengths(rt)  # lengths of rows in rt
      tf.Tensor([2, 0, 2, 1, 0])
      >>> rt.row_lengths(axis=2)  # lengths of axis=2 rows.
      <tf.RaggedTensor [[3, 1], [], [2, 1], [1], []]>
      ```
    """
    if self._cached_row_lengths is None:
      with ops.name_scope(name, "RaggedRowLengths", [self]):
        splits = self.row_splits
        self._cached_row_lengths = splits[1:] - splits[:-1]
    return self._cached_row_lengths

  def __str__(self):
    return "<tf.RaggedStructure %s>" % self.row_splits

  def __repr__(self):
    return "tf.RaggedStructure( %s)" % self._row_splits
