from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from phase8_correction_lib import LatentShiftCVAE, load_corrector  # noqa: E402
from phase8_train_latent_shift import save_checkpoint, strip_saved_batch  # noqa: E402


class LatentShiftCVAETest(unittest.TestCase):
    def make_model(self) -> LatentShiftCVAE:
        return LatentShiftCVAE(
            dim=32,
            hidden_dim=16,
            dropout=0.0,
            latent_dim=4,
        )

    def test_posterior_forward_backward_and_kl(self) -> None:
        model = self.make_model()
        hidden = torch.randn(2, 1, 4, 5, 32)
        target = torch.randn_like(hidden)

        prediction, stats = model(hidden, target_shift=target, return_stats=True)
        self.assertEqual(prediction.shape, hidden.shape)
        self.assertEqual(stats["kl_per_dim"].shape, (2, 4))
        self.assertTrue(torch.all(stats["kl_per_dim"] >= -1e-6))

        loss = (prediction - target).square().mean() + 0.01 * stats["kl_per_dim"].mean()
        loss.backward()
        self.assertIsNotNone(model.posterior_network[-1].weight.grad)
        self.assertIsNotNone(model.prior_network[-1].weight.grad)

    def test_prior_mean_is_deterministic_and_samples_have_expected_shape(self) -> None:
        model = self.make_model().eval()
        hidden = torch.randn(2, 1, 4, 5, 32)
        with torch.no_grad():
            first = model(hidden)
            second = model(hidden)
            samples = model.sample_shifts(hidden, 3)

        torch.testing.assert_close(first, second)
        self.assertEqual(samples.shape, (3, 2, 1, 4, 5, 32))
        self.assertFalse(torch.equal(samples[0], samples[1]))

    def test_saved_checkpoint_reloads_prior_mean_prediction(self) -> None:
        model = self.make_model().eval()
        hidden = torch.randn(2, 1, 4, 5, 32)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = pathlib.Path(tmp) / "cvae.pt"
            save_checkpoint(
                checkpoint,
                model=model,
                layer=27,
                dim=32,
                target_rms=1.5,
                hidden_dim=16,
                dropout=0.0,
                metrics={},
                train_args={"architecture": "cvae"},
                target_layer_spec="last",
                target_name="action",
            )
            loaded, payload = load_corrector(checkpoint, "cpu")

        self.assertEqual(payload["model_config"]["architecture"], "cvae")
        with torch.no_grad():
            torch.testing.assert_close(loaded(hidden), model(hidden))

    def test_capture_batch_is_removed_before_dataloader_batching(self) -> None:
        saved = torch.randn(1, 1, 4, 5, 32)
        self.assertEqual(strip_saved_batch(saved).shape, (1, 4, 5, 32))


if __name__ == "__main__":
    unittest.main()
