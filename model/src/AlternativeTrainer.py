from typing import TYPE_CHECKING, Any, Callable, Optional, Union
import torch
from torch import nn
from transformers import (
    Trainer,
)
from transformers.modeling_utils import unwrap_model

# torch.autograd.set_detect_anomaly(True)


class AlternatingTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        重写 training_step 方法来实现梯度交替更新。
        """
        loss = super().training_step(model, inputs)
        
        # train_which = 0 -> train gate_mask, zero gate's grad
        # train_which = 1 -> train gate, zero gate_mask's grad
        train_which = (self.state.global_step -1) % 2 

        unwrapped_model = unwrap_model(model)
        if train_which == 0:
            # print("zero gate grad")
            for layer in unwrapped_model.model.layers:
                if hasattr(layer.mlp, "gate") and hasattr(layer.mlp.gate, "weight") and layer.mlp.gate.weight.grad is not None:
                    layer.mlp.gate.weight.grad.zero_()
        else:
            # print("zero mask grad")
            for layer in unwrapped_model.model.layers:
                if hasattr(layer.mlp, "gate_mask") and layer.mlp.gate_mask.grad is not None:
                    layer.mlp.gate_mask.grad.zero_()

        return loss.detach()