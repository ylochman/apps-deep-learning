# Deep Learning Course. CNN implementation
## Theoretical solution
See `theory.pdf`.

## Practical solution
See `practice`.

Implementation of layers is in `simple_conv_net_func.py`.
You can check it by running `python test_layers.py`
(it calculates MSE for each PyTorch layer output vs. custom layer output)

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

### Training time
1 epoch training time for vector form  (run `time python simple_conv_net_train.py --custom --epochs 1 --single-batch`) is about **3.7 sec**:
```
real    0m3.083s
user    0m2.702s
sys     0m0.985s
```

For scalar form  (run `time python simple_conv_net_train.py --custom --scalar --epochs 1 --single-batch`) it's about **3 min 30.9 sec**:
```
real    3m30.357s
user    3m27.083s
sys     0m3.810s
```
Here, for both time experiments, a single batch of size 64 is passed.
