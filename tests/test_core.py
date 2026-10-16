"""Small numerical contracts for the released core operations."""
import unittest
import numpy as np
import torch
from melon_tss.regression import (
    fuse_huber_mse, fit_physics_preprocessor, fruit_median, huber_loss, mse_loss,
)
from melon_tss.segmentation import NetCLDiceLoss


class CoreContracts(unittest.TestCase):
    def test_median_is_after_branch_average(self):
        # Averaging branch medians would give 100; the actual order gives 50.
        average = fuse_huber_mse([0, 100, 100], [100, 0, 100])
        np.testing.assert_array_equal(fruit_median(average, ["a"] * 3), [50] * 3)

    def test_fold_preprocessing_does_not_refit_on_validation(self):
        rng = np.random.default_rng(2)
        training = rng.normal(size=(30, 20))
        training[0, 0] = np.nan
        pipe = fit_physics_preprocessor(training, n_components=4)
        before = pipe.named_steps["imputer"].statistics_.copy()
        prediction_features = pipe.transform(np.full((2, 20), 1e6))
        np.testing.assert_array_equal(pipe.named_steps["imputer"].statistics_, before)
        self.assertEqual(prediction_features.shape, (2, 4))
        self.assertAlmostEqual(before[0], np.nanmedian(training[:, 0]))

    def test_fixed_branch_losses(self):
        prediction = torch.tensor([0.0, 2.0])
        target = torch.tensor([1.0, 0.0])
        self.assertAlmostEqual(huber_loss(prediction, target).item(), 1.0)
        self.assertAlmostEqual(mse_loss(prediction, target).item(), 2.5)

    def test_ignored_pixels_do_not_contribute_to_net_loss_gradient(self):
        torch.manual_seed(3)
        logits = torch.randn(1, 4, 16, 16, requires_grad=True)
        target = torch.full((1, 16, 16), 2, dtype=torch.long)
        target[:, 7:9, 2:14] = 1
        target[:, :2] = 255
        loss = NetCLDiceLoss()(logits, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[:, :, :2].abs().max().item(), 0)
        self.assertGreater(logits.grad[:, :, 2:].abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
