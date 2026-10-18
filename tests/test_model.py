"""Bidirectional gradients, independent samples, serialization and angle loss."""
from dataclasses import asdict
from io import BytesIO
import unittest

import torch

from model import BiLSTMConfig, InverseBiLSTM, geometry_mse
from train import normalized_errors


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(23)
        self.config = BiLSTMConfig(patch_count=20, embedding_dim=16, hidden_size=8,
                                   recurrent_layers=2, mlp_dim=16)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_both_directions_receive_gradients_and_update(self):
        model = InverseBiLSTM(self.config)
        self.assertTrue(model.bilstm.bidirectional)
        self.assertEqual(model.encode(torch.rand(3, 1000)).shape, (3, 20, 16))
        before = {n: p.detach().clone() for n, p in model.bilstm.named_parameters()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        prediction = model(torch.rand(3, 1000))
        self.assertEqual(prediction.shape, (3, 7))
        geometry_mse(prediction, torch.rand(3, 7), 'square_periodic').backward()
        for name, p in model.bilstm.named_parameters():
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(p.grad.norm().item(), 0, name)
        optimizer.step()
        for name, p in model.bilstm.named_parameters():
            self.assertFalse(torch.equal(before[name], p), name)
        self.assertEqual(sum(isinstance(m, torch.nn.LSTM) for m in model.modules()), 1)
        self.assertFalse(any('attention' in n or 'resnet' in n for n, _ in model.named_modules()))

    def test_no_state_is_shared_between_spectra_and_input_forms_match(self):
        model = InverseBiLSTM(self.config).eval()
        x = torch.rand(3, 1000)
        with torch.inference_mode():
            batch = model(x)
            separate = torch.cat([model(row[None]) for row in x])
            channel_input = model(x[:, None])
        torch.testing.assert_close(batch, separate)
        torch.testing.assert_close(batch, channel_input, atol=0, rtol=0)

    def test_checkpoint_reload_has_identical_inference(self):
        model = InverseBiLSTM(self.config).eval()
        buffer = BytesIO()
        torch.save({'config': asdict(self.config), 'state': model.state_dict()}, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        restored = InverseBiLSTM(BiLSTMConfig.from_dict(saved['config'])).eval()
        restored.load_state_dict(saved['state'])
        x = torch.rand(2, 1000)
        with torch.inference_mode():
            torch.testing.assert_close(model(x), restored(x), atol=0, rtol=0)
        with self.assertRaises(ValueError):
            model(torch.rand(2, 999))
        with self.assertRaises(TypeError):
            BiLSTMConfig.from_dict({'attention_heads': 10})

    def test_periodic_angle_loss_matches_independent_metric(self):
        prediction = torch.full((2, 7), .5)
        target = prediction.clone()
        prediction[:, 5:] = .01
        target[:, 5:] = .99
        loss = geometry_mse(prediction, target, 'square_periodic')
        expected = 2 * .02**2 / 7
        self.assertAlmostEqual(loss.item(), expected, places=8)
        errors = normalized_errors(prediction.numpy(), target.numpy(), 'square_periodic')
        self.assertAlmostEqual(float((errors**2).mean()), expected, places=8)


if __name__ == '__main__':
    unittest.main()
