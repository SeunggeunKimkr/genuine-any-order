####################################################################################
### Copied from https://github.com/kuleshov-group/mdlm/blob/master/models/ema.py ###
####################################################################################

import torch
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf

def save_model_snapshot(
    ckpt_dir: str,
    model,
    cfg,
    epoch: int,
    global_step: int,
    val_loss: float = None,
    extra: dict = None,
):
    model_to_save = model.module if isinstance(model, DDP) else model

    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "val_loss": val_loss,
        "model_state_dict": model_to_save.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "is_ema_snapshot": False,
    }
    if extra is not None:
        checkpoint.update(extra)

    path = os.path.join(ckpt_dir, f"step={global_step}.pt")
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)
    return path

def save_ema_snapshot(
    ckpt_dir: str,
    model,
    ema,
    cfg,
    epoch: int,
    global_step: int,
    val_loss: float = None,
    extra: dict = None,
):
    model_to_save = model.module if isinstance(model, DDP) else model

    # swap in EMA weights
    ema.store(model_to_save.parameters())
    ema.copy_to(model_to_save.parameters())

    # build checkpoint
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "val_loss": val_loss,
        "model_state_dict": model_to_save.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "ema_state_dict": ema.state_dict(),
        "is_ema_snapshot": True,
    }
    if extra is not None:
        checkpoint.update(extra)

    # atomic-ish save (tmp -> rename)
    path = os.path.join(ckpt_dir, f"ema_step={global_step}.pt")
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)

    # restore original (non-EMA) weights
    ema.restore(model_to_save.parameters())
    return path


class ExponentialMovingAverage:
  """
  Maintains (exponential) moving average of a set of parameters.
  """

  def __init__(self, parameters, decay, use_num_updates=True):
    """
    Args:
        parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
        decay: The exponential decay.
        use_num_updates: Whether to use number of updates when computing
            averages.
    """
    if decay < 0.0 or decay > 1.0:
      raise ValueError('Decay must be between 0 and 1')
    self.decay = decay
    self.num_updates = 0 if use_num_updates else None
    self.shadow_params = [p.clone().detach()
                          for p in parameters if p.requires_grad]
    self.collected_params = []

  def move_shadow_params_to_device(self, device):
    self.shadow_params = [i.to(device) for i in self.shadow_params]

  def update(self, parameters):
    """
    Update currently maintained parameters.

    Call this every time the parameters are updated, such as the result of
    the `optimizer.step()` call.

    Args:
        parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
    """
    decay = self.decay
    if self.num_updates is not None:
      self.num_updates += 1
      decay = min(decay, (1 + self.num_updates) /
                  (10 + self.num_updates))
    one_minus_decay = 1.0 - decay
    with torch.no_grad():
      parameters = [p for p in parameters if p.requires_grad]
      for s_param, param in zip(self.shadow_params, parameters):
        s_param.sub_(one_minus_decay * (s_param - param))

  def copy_to(self, parameters):
    """
    Copy current parameters into given collection of parameters.

    Args:
        parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
    """
    parameters = [p for p in parameters if p.requires_grad]
    for s_param, param in zip(self.shadow_params, parameters):
      if param.requires_grad:
        param.data.copy_(s_param.data)

  def store(self, parameters):
    """
    Save the current parameters for restoring later.

    Args:
        parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
    """
    self.collected_params = [param.clone() for param in parameters]

  def restore(self, parameters):
    """
    Restore the parameters stored with the `store` method.
    Useful to validate the model with EMA parameters without affecting the
    original optimization process. Store the parameters before the
    `copy_to` method. After validation (or model saving), use this to
    restore the former parameters.

    Args:
        parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
    """
    for c_param, param in zip(self.collected_params, parameters):
      param.data.copy_(c_param.data)

  def state_dict(self):
    return dict(decay=self.decay,
                num_updates=self.num_updates,
                shadow_params=self.shadow_params)

  def load_state_dict(self, state_dict):
    self.decay = state_dict['decay']
    self.num_updates = state_dict['num_updates']
    self.shadow_params = state_dict['shadow_params']