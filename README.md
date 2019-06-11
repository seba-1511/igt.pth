# Implicit Gradient Transport in PyTorch

This repository is the original code for the experiments in: [https://arxiv.org/abs/1906.03532](https://arxiv.org/abs/1906.03532).

For more information, please refer to our [paper](https://arxiv.org/abs/1906.03532) or our [website](http://seba1511.net/projects/igt/).

## Citation

You can cite the algorithms implemented in this repository by citing the following paper.

~~~
@misc{arnold2019reducing,
    title={Reducing the variance in online optimization by transporting
           past gradients},
    author={Sebastien M. R. Arnold,
            Pierre-Antoine Manzagol,
            Reza Babanezhad,
            Ioannis Mitliagkas,
            Nicolas Le Roux},
    year={2019},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
~~~

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

# Note

The ITA family of algorithms (such as `Heavyball-ITA` in the paper) are implemented as `torch_igt.ITA(params, opt)`.
