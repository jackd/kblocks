import numpy as np
import tensorflow as tf

import kblocks.ops.sparse as sparse_ops


class SparseOpsTest(tf.test.TestCase):
    def test_ragged_to_sparse_indices(self):
        values = tf.constant([0, 1, 2, 0, 1, 0, 1, 2, 3], dtype=tf.int64)
        rs0 = tf.constant([0, 3, 5, 9])
        rs1 = tf.constant([0, 2, 3])
        offset = tf.constant([0, 10], dtype=tf.int64)

        rt = tf.RaggedTensor.from_row_splits(values, rs0)
        rt = tf.RaggedTensor.from_row_splits(rt, rs1)

        b, i, j = self.evaluate(sparse_ops.ragged_to_sparse_indices(rt, offset))
        np.testing.assert_equal(b, [0, 0, 0, 0, 0, 1, 1, 1, 1])
        np.testing.assert_equal(i, [0, 0, 0, 1, 1, 2, 2, 2, 2])
        np.testing.assert_equal(j, [0, 1, 2, 0, 1, 10, 11, 12, 13])

    def test_block_diagonalize(self):
        b0 = [0, 0, 0, 0]
        i0 = [0, 2, 2, 3]
        j0 = [2, 5, 6, 8]

        b1 = [1, 1, 1]
        i1 = [1, 3, 3]
        j1 = [0, 4, 9]

        shape = tf.constant([2, 4, 10], dtype=tf.int64)

        b = tf.constant(np.concatenate([b0, b1]), dtype=tf.int64)
        i = tf.constant(np.concatenate([i0, i1]), dtype=tf.int64)
        j = tf.constant(np.concatenate([j0, j1]), dtype=tf.int64)

        (i, j), shape = sparse_ops.block_diagonalize_sparse((b, i, j), shape)

        expected_shape = [8, 20]
        expected_i = [0, 2, 2, 3, 5, 7, 7]
        expected_j = [2, 5, 6, 8, 10, 14, 19]
        shape, i, j = self.evaluate((shape, i, j))
        np.testing.assert_equal(shape, expected_shape)
        np.testing.assert_equal(i, expected_i)
        np.testing.assert_equal(j, expected_j)

    def test_unstack(self):
        values = tf.random.uniform(shape=(11,), dtype=tf.float32)
        indices = tf.constant(
            [
                [0, 0, 2],
                [0, 0, 5],
                [0, 1, 1],
                [0, 1, 2],
                [0, 2, 4],
                [1, 0, 2],
                [1, 0, 5],
                [1, 1, 1],
                [1, 2, 2],
                [1, 2, 4],
                [1, 2, 5],
            ],
            dtype=tf.int64,
        )
        dense_shape = tf.constant((2, 3, 6), dtype=tf.int64)
        st = tf.SparseTensor(indices, values, dense_shape)
        unstacked = sparse_ops.unstack(st, axis=0)
        self.assertEqual(len(unstacked), 2)
        np_values, unstacked = self.evaluate((values, unstacked))

        np.testing.assert_equal(
            unstacked[0].indices, [[0, 2], [0, 5], [1, 1], [1, 2], [2, 4],]
        )
        np.testing.assert_equal(
            unstacked[1].indices, [[0, 2], [0, 5], [1, 1], [2, 2], [2, 4], [2, 5],]
        )

        np.testing.assert_allclose(unstacked[0].values, np_values[:5])
        np.testing.assert_allclose(unstacked[1].values, np_values[5:])
        for u in unstacked:
            np.testing.assert_equal(u.dense_shape, [3, 6])

        unstacked = sparse_ops.unstack(st, axis=1)
        np_values, unstacked = self.evaluate((values, unstacked))

        np.testing.assert_equal(unstacked[0].indices, [[0, 2], [0, 5], [1, 2], [1, 5]])
        np.testing.assert_equal(unstacked[1].indices, [[0, 1], [0, 2], [1, 1],])
        np.testing.assert_equal(unstacked[2].indices, [[0, 4], [1, 2], [1, 4], [1, 5],])

        np.testing.assert_allclose(unstacked[0].values, np_values[[0, 1, 5, 6]])
        np.testing.assert_allclose(unstacked[1].values, np_values[[2, 3, 7]])
        np.testing.assert_allclose(unstacked[2].values, np_values[[4, 8, 9, 10]])
        for u in unstacked:
            np.testing.assert_equal(u.dense_shape, (2, 6))

    def test_remove_dim(self):
        values = tf.random.uniform(shape=(11,), dtype=tf.float32)
        indices = tf.constant(
            [
                [0, 0, 2],
                [0, 0, 5],
                [0, 1, 1],
                [0, 1, 2],
                [0, 2, 4],
                [1, 0, 2],
                [1, 0, 5],
                [1, 1, 1],
                [1, 2, 2],
                [1, 2, 4],
                [1, 2, 5],
            ],
            dtype=tf.int64,
        )
        dense_shape = tf.constant((2, 3, 6), dtype=tf.int64)
        st = tf.SparseTensor(indices, values, dense_shape)
        # st = sparse_ops.remove_dim(st, axis=0)
        st = tf.keras.layers.Lambda(sparse_ops.remove_dim, arguments=dict(axis=0))(st)
        values, st = self.evaluate((values, st))
        np.testing.assert_allclose(st.values, values)
        np.testing.assert_equal(st.dense_shape, (3, 6))
        np.testing.assert_equal(
            st.indices,
            [
                [0, 2],
                [0, 5],
                [1, 1],
                [1, 2],
                [2, 4],
                [0, 2],
                [0, 5],
                [1, 1],
                [2, 2],
                [2, 4],
                [2, 5],
            ],
        )


if __name__ == "__main__":
    tf.test.main()
