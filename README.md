提供Condition Quantile Estimation的代码，如下

---

## 代码实现

```python
# TensorFlow 实现
import tensorflow as tf

class ConditionQuantileEstimation:
    def __init__(self, quan_bins):
        self.quan_bins = quan_bins

    def get_quans(self, feature, name=''):
        quan_pred = simple_dense_network(
            feature, [128, self.quan_bins - 1], name + 'quan_s', name + 'quansr_h{}_params', act=tf.nn.relu
        )
        quan_pred = tf.math.cumsum(quan_pred, axis=-1)
        return quan_pred

    def get_singal_loss(self, y_true, y_pred, tau=0.5):
        """
        计算 pinball 损失函数。
        """
        error = y_true - y_pred
        pinball_loss = tf.reduce_sum(tf.maximum(tau * error, (tau - 1) * error))
        return pinball_loss

    def get_loss(self, y_true, quan_pred):
        taus = tf.range(self.quan_bins - 1, dtype=tf.float32) / self.quan_bins + 1 / self.quan_bins
        taus = tf.expand_dims(taus, axis=0)
        loss_quan = self.get_singal_loss(y_true, quan_pred, taus)
        return loss_quan

# PyTorch 实现
import torch

def quantile_loss(yhat, y):
    """
    计算 PyTorch 版本的分位数损失。
    """
    N, M = yhat.shape
    quantiles = torch.arange(1, M + 1, dtype=yhat.dtype, device=yhat.device) / (M + 1)
    quantiles = quantiles.unsqueeze(0).expand(N, M)
    y = y.view(N, 1).expand(-1, M)
    
    l1 = (y - yhat) * quantiles * (y > yhat).float()
    l2 = (yhat - y) * (1 - quantiles) * (y < yhat).float()
    l = (l1 + l2).sum(dim=-1)
    
    return l.mean()
```

---

## 参考
- TensorFlow 官方文档：[https://www.tensorflow.org](https://www.tensorflow.org)
- PyTorch 官方文档：[https://pytorch.org](https://pytorch.org)


