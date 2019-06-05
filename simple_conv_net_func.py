from __future__ import print_function
import torch


def diff_mse(x, y):
    x_vec = x.view(1, -1).squeeze()
    y_vec = y.view(1, -1).squeeze()
    return torch.mean(torch.pow((x_vec - y_vec), 2)).item()

def conv2d_scalar(x_in, conv_weight, conv_bias, device):
    #
    # Add your code here
    #
    pass


def conv2d_vector(x_in, conv_weight, conv_bias, device):
    """
    Args:
        x_in: [N, C_in, S_in, S_in] tensor
        conv_weight: [C_out, C_in, K, K] tensor
        conv_bias: [C_out] tensor
        device: 'cpu' or 'cuda'
    Returns:
        x_out: [N, C_out, S_out, S_out] tensor where S_out = S_in - K + 1
    """
    # Assertions
    assert x_in.ndimension() == 4
    assert conv_weight.ndimension() == 4
    assert conv_bias.ndimension() == 1
    assert x_in.size()[1] == conv_weight.size()[1]
    assert conv_bias.size()[0] == conv_weight.size()[0]
    assert x_in.size()[2] == x_in.size()[3]
    assert conv_weight.size()[2] == conv_weight.size()[3]

    # Shapes
    N, C_in, S_in, S_in = x_in.size()
    C_out, _, K, K = conv_weight.size()
    S_out = S_in - K + 1

    x_out = torch.zeros([N, C_out, S_out, S_out])

    # Moving to device
    x_in = x_in.to(device)
    x_out = x_out.to(device)
    conv_weight = conv_weight.to(device)
    conv_bias = conv_bias.to(device)

    # Convolution loop
    for i in range(0, S_out, 1):
        for j in range(0, S_out, 1):
            # x_in_patch: [N, C_in, K, K]
            x_in_patch = x_in[:, :, i:i+K, j:j+K].contiguous()
            # x_out_patch: [N, C_out, 1, 1] = [N, 1, 1, C_in * K * K] @ [C_out, C_in * K * K, 1]
            x_out[:,:,i,j] = torch.matmul(
                x_in_patch.view(N, 1, 1, C_in * K * K),
                conv_weight.view(C_out, C_in * K * K, 1))[:,:,0,0] + conv_bias
    return x_out


def im2col(X, kernel_size, device):
    #
    # Add your code here
    #
    pass


def conv_weight2rows(conv_weight):
    #
    # Add your code here
    #
    pass


def pool2d_scalar(a, device):
    #
    # Add your code here
    #
    pass


def pool2d_vector(x_in, device):
    """
    Args:
        x_in: [N, C_in, S_in, S_in] tensor
        device: 'cpu' or 'cuda'
    Returns:
        x_out: [N, C_in, S_out, S_out] tensor where S_out = S_in // 2
    """
    # Assertions
    assert x_in.ndimension() == 4
    assert x_in.size()[2] == x_in.size()[3]
    assert x_in.size()[2] % 2 == 0

    # Shapes
    N, C_in, S_in, S_in = x_in.size()
    S_out = S_in // 2
    K = 2
    stride = 2

    x_out = torch.zeros([N, C_in, S_out, S_out])

    # Moving to device
    x_in = x_in.to(device)
    x_out = x_out.to(device)

    # Pooling loop
    for i in range(0, S_out, stride):
        for j in range(0, S_out, stride):
            # x_in_patch: [N, C_in, K, K]
            x_in_patch = x_in[:, :, i:i+K, j:j+K].contiguous()
            # x_out_patch: [N, C_out, 1, 1] = [N, 1, 1, C_in * K * K] @ [C_out, C_in * K * K, 1]
            x_out[:,:,i,j] = x_in_patch.view(N, C_in, K * K).max(dim=2)[0]
    return x_out


def relu_scalar(a, device):
    #
    # Add your code here
    #
    pass


def relu_vector(a, device):
    #
    # Add your code here
    #
    pass


def reshape_vector(a, device):
    #
    # Add your code here
    #
    pass


def reshape_scalar(a, device):
    #
    # Add your code here
    #
    pass

def fc_layer_scalar(a, weight, bias, device):
    #
    # Add your code here
    #
    pass


def fc_layer_vector(x_in, weight, bias, device):
    """
    Args:
        x_in: [N, C_in] tensor
        weight: [C_out, C_in] tensor
        bias: [C_out] tensor
        device: 'cpu' or 'cuda'
    Returns:
        x_out: [N, C_out] tensor
    """
    # Assertions
    assert x_in.ndimension() == 2
    assert weight.ndimension() == 2
    assert bias.ndimension() == 1
    assert x_in.size()[1] == weight.size()[1]
    assert bias.size()[0] == weight.size()[0]

    # Shapes
    N, C_in = x_in.size()
    C_out, _ = weight.size()

    # Moving to device
    x_in = x_in.to(device)
    weight = weight.to(device)
    bias = bias.to(device)

    # Fully connected layer operation
    # x_out: [N, C_out] = [N, 1, 1, C_in] @ [C_out, C_in, 1]
    x_out = torch.matmul(x_in.contiguous().view(N, 1, 1, C_in),
                        weight.view(C_out, C_in, 1))[:,:,0,0] + bias
    return x_out
