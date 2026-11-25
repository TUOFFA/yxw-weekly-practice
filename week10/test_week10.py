"""对第十周最关键的结构性质做快速单元测试。"""

import unittest

import torch

from models import AttentionHead, TransformerEncoderPredictor


class TransformerStructureTests(unittest.TestCase):
    """验证参数复制、因果遮罩和编码器形状。"""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.inputs = torch.randn(1, 24, 16)

    def test_copied_heads_match(self) -> None:
        head1 = AttentionHead()
        head2 = AttentionHead()
        head2.load_state_dict(head1.state_dict())
        weights1, context1 = head1(self.inputs)
        weights2, context2 = head2(self.inputs)
        self.assertTrue(torch.equal(weights1, weights2))
        self.assertTrue(torch.equal(context1, context2))

    def test_causal_mask_blocks_future_change(self) -> None:
        head = AttentionHead()
        changed = self.inputs.clone()
        changed[0, 17, :] += 1
        weights, original_context = head(self.inputs, use_causal_mask=True)
        _, changed_context = head(changed, use_causal_mask=True)
        self.assertEqual(float(weights[0, 5, 17].item()), 0.0)
        self.assertTrue(torch.equal(original_context[0, 5], changed_context[0, 5]))

    def test_encoder_keeps_rows_until_prediction(self) -> None:
        model = TransformerEncoderPredictor()
        result = model.inspect_forward(torch.randn(1, 24, 1))
        for name in ["X", "M", "H", "F", "E"]:
            self.assertEqual(tuple(result[name].shape), (1, 24, 16))
        self.assertEqual(tuple(result["last_row"].shape), (1, 16))
        self.assertEqual(tuple(result["prediction"].shape), (1, 1))


if __name__ == "__main__":
    unittest.main()
