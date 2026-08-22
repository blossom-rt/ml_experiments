"""part2 CNN 模型测试

覆盖：模型参数量、前向输出形状、eval 模式确定性、Dropout 在训练模式下生效。
"""
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'part2'))
from cnn_training import ImprovedCNN


def test_parameter_count():
    model = ImprovedCNN()
    n = sum(p.numel() for p in model.parameters())
    assert n == 871530, f"参数量不符: {n}"


def test_forward_pass_shape():
    model = ImprovedCNN()
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 10)


def test_eval_mode_deterministic():
    model = ImprovedCNN()
    model.eval()
    x = torch.randn(4, 1, 28, 28)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2)


def test_dropout_active_in_train():
    """训练模式下 Dropout 应生效，两次前向结果不同"""
    model = ImprovedCNN()
    model.train()
    x = torch.randn(4, 1, 28, 28)
    out1 = model(x)
    out2 = model(x)
    assert not torch.allclose(out1, out2)