import unittest

import matplotlib.pyplot as plt
import pandas as pd

from robust_dm_factor_allocation.plots import plot_wealth, plot_weights


class PlotTest(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_plot_wealth_supports_default_and_supplied_axes(self):
        returns = pd.Series([0.01, -0.02, 0.03])
        default_axis = plot_wealth(returns, log_scale=True)
        self.assertEqual(default_axis.get_ylabel(), "Wealth index")
        self.assertEqual(default_axis.get_yscale(), "log")

        _, supplied_axis = plt.subplots()
        self.assertIs(plot_wealth(returns, ax=supplied_axis), supplied_axis)

    def test_plot_weights_validates_and_draws_a_stack(self):
        with self.assertRaises(ValueError):
            plot_weights(pd.DataFrame())
        with self.assertRaises(ValueError):
            plot_weights([0.5, 0.5])

        weights = pd.DataFrame({"a": [0.5, 0.4], "b": [0.5, 0.6]})
        default_axis = plot_weights(weights)
        self.assertEqual(default_axis.get_ylim(), (0.0, 1.0))
        _, supplied_axis = plt.subplots()
        self.assertIs(plot_weights(weights, ax=supplied_axis), supplied_axis)


if __name__ == "__main__":
    unittest.main()
