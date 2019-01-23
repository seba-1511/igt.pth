# igt.pth
Implicit Gradient Transport in PyTorch

# Installation

```
pip install -e .
```

# Usage

See `tests/` folder for more examples.

```python
import torch.optim as optim
from torch_igt import IGTransporter

opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
opt = IGTransporter(model.parameters(), opt)

# Compute a single optimization step
opt.train()  # Ensures parameters are set to the transported ones
loss = L(model(X_train), y_train)
opt.zero_grad()
loss.backward()
opt.step()

# Reverts parameters to the true ones
opt.eval()
loss = L(model(X_test), y_test)
```

# Note to Reviewers

The ATA family of algorithms (such as `Heavyball-ATA` in the paper) are implemented as `torch_igt.NCIGT(params, opt)`.
