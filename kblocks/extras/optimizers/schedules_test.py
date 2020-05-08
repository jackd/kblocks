from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from kblocks.extras.optimizers import schedules


class SchedulesTest(tf.test.TestCase):
    def test_exponential_decay(self):
        self.assertAllClose(
            schedules.exponential_decay(5, 1.0, 2.0, 0.5, staircase=True), 1 / 4
        )
        self.assertAllClose(
            schedules.exponential_decay(5, 1.0, 2.0, 0.5, staircase=False), 2 ** (-2.5)
        )
        self.assertAllClose(
            schedules.exponential_decay(1000, 1.0, 2, 0.5, min_value=0.1), 0.1
        )
        self.assertAllClose(
            schedules.exponential_decay_towards(100, 0, 1, 0.5, asymptote=2), 2
        )
        self.assertAllClose(
            schedules.exponential_decay_towards(
                100, 1.5, 1, 0.5, asymptote=2, clip_value=1.9
            ),
            1.9,
        )
        self.assertAllClose(
            schedules.exponential_decay_towards(100, 2, 1, 0.5, asymptote=1), 1.0
        )
        self.assertAllClose(
            schedules.exponential_decay_towards(100, 2, 1, 0.5, asymptote=1.1), 1.1
        )

    def test_cosine_annealing(self):
        max_value = 10.0
        min_value = 1.0
        steps_per_restart = 100.0
        kwargs = dict(
            max_value=max_value,
            min_value=min_value,
            steps_per_restart=steps_per_restart,
        )
        for step, expected in (
            (0, max_value),
            (steps_per_restart, max_value),
            (steps_per_restart - 1, min_value),
        ):
            self.assertAllClose(
                self.evaluate(schedules.cosine_annealing(step, **kwargs)), expected
            )


if __name__ == "__main__":
    tf.test.main()
