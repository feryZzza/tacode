"""forecast_residual 的因果性与时间对齐。

在线门控在时刻 u 读到的残差只能由 u 时刻已观测到的量构成。forecast[t] 的目标是
input_norm[t+h]，所以该残差最早在 t+h 可得，必须写在 t+h 位置；开头 h 帧无残差。
"""

import unittest

import torch

from reliability.metrics import forecast_residual


class ForecastResidualCausalityTest(unittest.TestCase):
	def test_horizon_one_publishes_at_target_arrival(self) -> None:
		# forecast[t] == input_norm[t] 时，残差 = (input[t] - input[t+1])^2，发布在 t+1。
		input_norm = torch.tensor([[[0.0, 1.0, 3.0, 6.0]]])
		forecast = input_norm.clone()

		residual = forecast_residual(forecast, input_norm, horizon=1)

		self.assertEqual(residual.shape, input_norm.shape)
		self.assertEqual(residual[0, 0, 0].item(), 0.0)
		expected = [0.0, 1.0, 4.0, 9.0]
		for index, value in enumerate(expected):
			self.assertAlmostEqual(residual[0, 0, index].item(), value, places=5)

	def test_horizon_two_zeroes_first_two_frames(self) -> None:
		input_norm = torch.arange(6, dtype=torch.float32).reshape(1, 1, 6)
		forecast = torch.zeros_like(input_norm)

		residual = forecast_residual(forecast, input_norm, horizon=2)

		self.assertEqual(residual.shape, input_norm.shape)
		self.assertEqual(residual[0, 0, 0].item(), 0.0)
		self.assertEqual(residual[0, 0, 1].item(), 0.0)
		# 位置 u 的残差由 forecast[u-2]=0 与 input_norm[u]=u 构成，即 u^2。
		for u in range(2, 6):
			self.assertAlmostEqual(residual[0, 0, u].item(), float(u * u), places=5)

	def test_residual_at_time_u_ignores_inputs_after_u(self) -> None:
		"""改动 input_norm[u+1:] 不能影响时刻 u 的残差（无未来泄漏）。"""
		torch.manual_seed(0)
		forecast = torch.randn(2, 3, 12)
		input_norm = torch.randn(2, 3, 12)
		cut = 7

		baseline = forecast_residual(forecast, input_norm, horizon=1)
		perturbed_input = input_norm.clone()
		perturbed_input[..., cut + 1:] += 5.0
		perturbed = forecast_residual(forecast, perturbed_input, horizon=1)

		torch.testing.assert_close(baseline[..., : cut + 1], perturbed[..., : cut + 1])

	def test_feature_groups_preserve_time_length(self) -> None:
		forecast = torch.randn(2, 4, 9)
		input_norm = torch.randn(2, 4, 9)
		groups = {"a": [0, 1], "b": [2, 3]}

		residual = forecast_residual(forecast, input_norm, horizon=3, feature_groups=groups)

		self.assertEqual(residual.shape, (2, 1, 9))
		self.assertTrue(torch.all(residual[..., :3] == 0))

	def test_sequence_shorter_than_horizon_is_all_zero(self) -> None:
		forecast = torch.randn(1, 2, 2)
		input_norm = torch.randn(1, 2, 2)

		residual = forecast_residual(forecast, input_norm, horizon=2)

		self.assertEqual(residual.shape, (1, 1, 2))
		self.assertTrue(torch.all(residual == 0))


if __name__ == "__main__":
	unittest.main()
