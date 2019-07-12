import torch.nn.functional as F
from simple_conv_net_train import SimpleConvNet
from simple_conv_net_func import *

def test_equality(x, y):
    assert diff_mse(x, y) < 1e-6

def compare_with_pytorch(fn='vector'):
    model = SimpleConvNet('cpu', True)
    x_in = torch.rand([4,1,28,28])
    conv_weight = model.conv_layer.weight
    conv_bias = model.conv_layer.bias
    weight = model.fc_layer1.weight
    bias = model.fc_layer1.bias
    device = 'cpu'

    if fn == 'scalar':
        conv2d = conv2d_scalar
        pool2d = pool2d_scalar
        reshape = reshape_scalar
        fc_layer = fc_layer_scalar
        relu = relu_scalar
    else:
        conv2d = conv2d_vector
        pool2d = pool2d_vector
        reshape = reshape_vector
        fc_layer = fc_layer_vector
        relu = relu_vector

    try:
        print('> {} form'.format(fn))
        print('Checking conv..')
        test_equality(model.conv_layer(x_in), conv2d(x_in, conv_weight, conv_bias, device))
        x_in = model.conv_layer(x_in)

        print('Checking maxpool..')
        test_equality(F.max_pool2d(x_in, 2, 2), pool2d(x_in, device))
        x_in = F.max_pool2d(x_in, 2, 2)

        print('Checking reshape..')
        test_equality(x_in.view(-1, 20*12*12), reshape(x_in, device))
        x_in = x_in.view(-1, 20*12*12)

        print('Checking fc..')
        test_equality(model.fc_layer1(x_in), fc_layer(x_in, weight, bias, device))
        x_in = model.fc_layer1(x_in)

        print('Checking relu..')
        test_equality(F.relu(x_in), relu(x_in, device))

        print('Perfect! ^__^')
    except:
        print('Something went wrong.. :(')
    
if __name__ == "__main__":
    compare_with_pytorch('scalar')
    compare_with_pytorch('vector')