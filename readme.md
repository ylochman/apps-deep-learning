# Deep Learning Course
## Understanding of CNNs
### Theoretical solution
See `theory.pdf`.

### Practical solution
See `practice`.
Implementation of layers is in `simple_conv_net_func.py`.
You can check it by running `python test.py`
(it calculates MSE for each PyTorch implemented layer output vs. custom layer output)

### Results
After training using PyTorch layers (run `python simple_conv_net_train.py`) I got:
```
Test set: Average loss: -0.9852, Accuracy: 9871/10000 (98.71%)
```

After training with custom layers (run `python simple_conv_net_train.py --custom`) I got:
```
Test set: Average loss: -0.9852, Accuracy: 9876/10000 (98.76%)
```

The training details for both NNs are the same and following:
* 20 epochs
* batch size 64
* lr 0.01
* SGD momentum 0.5
