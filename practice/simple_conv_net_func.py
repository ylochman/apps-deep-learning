from __future__ import print_function
import torch


def diff_mse(x, y):
    x_vec = x.view(1, -1).squeeze()
    y_vec = y.view(1, -1).squeeze()
    return torch.mean(torch.pow((x_vec - y_vec), 2)).item()


def im2col(x_in, K, stride=1, keep_channels=False):
    """
    Args:
        x_in: [N, C_in, S_in, S_in] tensor
        K: int
        stride: int
        keep_channels: bool
    Returns:
        x_in_cols: [N, C_in * K * K, S_out * S_out] tensor
                    if keep_channels is False or
                   [N, C_in, K * K, S_out * S_out] tensor
                    if keep_channels is True,
                    where S_out = floor((S_in - K) / stride + 1)

    """
    N, C_in, S_in, S_in = x_in.size()
    S_out = int((S_in - K) / stride + 1)

    if keep_channels:
        i0 = torch.arange(K).repeat(1,K).view(-1, K).transpose(0,1).contiguous()
        i1 = torch.arange(0, S_in, stride).repeat(1,S_out).view(-1,S_out).transpose(0,1).contiguous()
        i = i0.view(-1, 1) + i1.view(1, -1)
        j0 = torch.arange(K).repeat(K)
        j1 = torch.arange(0, S_in, stride).repeat(S_out)
        j = j0.view(-1, 1) + j1.view(1, -1)
        return x_in[:, :, i, j]
    else:
        i0 = torch.arange(K).repeat(1,K).view(-1, K).transpose(0,1).repeat(1,C_in)
        i1 = torch.arange(S_out).repeat(1,S_out).view(-1,S_out).transpose(0,1).contiguous()
        i = i0.view(-1, 1) + i1.view(1, -1)
        j0 = torch.arange(K).repeat(K * C_in)
        j1 = torch.arange(S_out).repeat(S_out)
        j = j0.view(-1, 1) + j1.view(1, -1)
        k = torch.arange(C_in).repeat(1,K*K).view(-1, K*K).transpose(0,1).view(-1, 1)
        return x_in[:, k, i, j]

def conv_weight2rows(conv_weight):
    """
    Args:
        conv_weight: [C_out, C_in, K, K] tensor
    Returns:
        conv_weight_rows: [C_out, S] tensor where S = C_in * K * K
    """
    C_out, C_in, K, K = conv_weight.size()
    conv_weight_rows = conv_weight.view(C_out, C_in * K * K)
    return conv_weight_rows


def assert_conv2d(x_in, conv_weight, conv_bias):
    assert x_in.ndimension() == 4
    assert conv_weight.ndimension() == 4
    assert conv_bias.ndimension() == 1
    assert x_in.size()[1] == conv_weight.size()[1]
    assert conv_bias.size()[0] == conv_weight.size()[0]
    assert x_in.size()[2] == x_in.size()[3]
    assert conv_weight.size()[2] == conv_weight.size()[3]

def conv2d_scalar(x_in, conv_weight, conv_bias, device='cpu'):
    """
    Args:
        x_in: [N, C_in, S_in, S_in] tensor
        conv_weight: [C_out, C_in, K, K] tensor
        conv_bias: [C_out] tensor
        device: 'cpu' or 'cuda'
    Returns:
        x_out: [N, C_out, S_out, S_out] tensor where S_out = S_in - K + 1
    """
    assert_conv2d(x_in, conv_weight, conv_bias)
    # Shapes
    N, C_in, S_in, S_in = x_in.size()
    C_out, _, K, K = conv_weight.size()
    S_out = S_in - K + 1
    # Moving to device
    x_in = x_in.to(device)
    conv_weight = conv_weight.to(device)
    conv_bias = conv_bias.to(device)
    # Convolution
    x_out = torch.zeros([N, C_out, S_out, S_out]).to(device)
    for c in range(0, C_out):
        for i in range(0, S_out):
            for j in range(0, S_out):
                x_in_patch = x_in[:, :, i:i+K, j:j+K].contiguous()
                x_out[:,c,i,j] = (x_in_patch * conv_weight[c]).sum((1,2,3)) + conv_bias[c]
    return x_out

def conv2d_vector(x_in, conv_weight, conv_bias, device='cpu'):
    """
    Args:
        x_in: [N, C_in, S_in, S_in] tensor
        conv_weight: [C_out, C_in, K, K] tensor
        conv_bias: [C_out] tensor
        device: 'cpu' or 'cuda'
    Returns:
        x_out: [N, C_out, S_out, S_out] tensor where S_out = S_in - K + 1
    """
    assert_conv2d(x_in, conv_weight, conv_bias)
    # Shapes
    N, C_in, S_in, S_in = x_in.size()
    C_out, _, K, K = conv_weight.size()
    S_out = S_in - K + 1
    # Moving to device
    x_in = x_in.to(device)
    conv_weight = conv_weight.to(device)
    conv_bias = conv_bias.to(device)
    # Convolution
    x_in_cols = im2col(x_in, K) 
    conv_weight_rows = conv_weight2rows(conv_weight)
    # [C_out, S] @ [N, S, S_out * S_out] -> [N, C_out, S_out * S_out] -> [N, C_out, S_out, S_out]
    x_out = torch.matmul(conv_weight_rows, x_in_cols) + conv_bias.view(1, C_out, 1)
    return x_out.view(N, C_out, S_out, S_out)


def assert_pool2d(x_in):
    assert x_in.ndimension() == 4
    assert x_in.size()[2] == x_in.size()[3]
    assert x_in.size()[2] % 2 == 0

def pool2d_scalar(x_in, device='cpu'):
    """
    Args:
        x_in: [N, C_in, S_in, S_in] tensor
        device: 'cpu' or 'cuda'
    Returns:
        x_out: [N, C_in, S_out, S_out] tensor where S_out = S_in // 2
    """
    assert_pool2d(x_in)
    # Shapes
    N, C_in, S_in, S_in = x_in.size()
    K = 2
    stride = 2
    S_out = S_in // stride
    # Moving to device
    x_in = x_in.to(device)
    # Pooling loop
    x_out = torch.zeros([N, C_in, S_out, S_out]).to(device)
    for i in range(0, S_in, stride):
        for j in range(0, S_in, stride):
            x_out[:,:,i//stride,j//stride] = x_in[:, :, i:i+K, j:j+K].max(3).values.max(2).values
    return x_out

def pool2d_vector(x_in, device='cpu'):
    """
    Args:
        x_in: [N, C_in, S_in, S_in] tensor
        device: 'cpu' or 'cuda'
    Returns:
        x_out: [N, C_in, S_out, S_out] tensor where S_out = S_in // 2
    """
    assert_pool2d(x_in)
    # Shapes
    N, C_in, S_in, S_in = x_in.size()
    K = 2
    stride = 2
    S_out = S_in // stride
    # Moving to device
    x_in = x_in.to(device)
    # Pooling
    x_in_cols = im2col(x_in, K, stride, keep_channels=True)
    # [N, C_in, K * K, S_out * S_out] -> [N, C_in, S_out * S_out] -> [N, C_in, S_out, S_out]
    x_out = x_in_cols.max(2).values
    return x_out.view(N, C_in, S_out, S_out)


def relu_scalar(x_in, device='cpu'):
    """
    Args:
        x_in: any size tensor
        device: 'cpu' or 'cuda'
    Returns:
        x_out:  tensor where S_out = S_in // 2
    """
    sizes = x_in.size()
    x_in = x_in.view(-1).to(device)
    x_out = torch.zeros_like(x_in)
    for i in range(x_out.size(0)):
        x_out[i] = torch.max(x_in[i], torch.zeros(1))
    return x_out.view(sizes)

def relu_vector(x_in, device='cpu'):
    """
    Args:
        x_in: any size tensor
        device: 'cpu' or 'cuda'
    Returns:
        x_out:  tensor where S_out = S_in // 2
    """
    x_in = x_in.to(device)
    return torch.max(x_in, torch.zeros_like(x_in))
    

def reshape_scalar(x_in, device='cpu'):
    """
    Args:
        x_in: [N, C_in, S_in, S_in] tensor
        device: 'cpu' or 'cuda'
    Returns:
        x_out: [N, C_in * S_in * S_in] tensor
    """
    assert x_in.ndimension() == 4
    N, C_in, S_in, _ = x_in.size()
    L = C_in * S_in * S_in
    x_in = x_in.to(device)
    x_out = torch.zeros([N, L]).to(device)
    for c in range(C_in):
        for i in range(S_in):
            for j in range(S_in):
                l = c * S_in * S_in + i * S_in + j
                x_out[:, l] = x_in[:, c, i, j]
    return x_out

def reshape_vector(x_in, device='cpu'):
    """
    Args:
        x_in: [N, C_in, S_in, S_in] tensor
        device: 'cpu' or 'cuda'
    Returns:
        x_out: [N, C_in * S_in * S_in] tensor
    """
    assert x_in.ndimension() == 4
    N = x_in.size()[0]
    x_in = x_in.to(device)
    return x_in.view(N, -1)


def assert_fc(x_in, weight, bias):
    assert x_in.ndimension() == 2
    assert weight.ndimension() == 2
    assert bias.ndimension() == 1
    assert x_in.size()[1] == weight.size()[1]
    assert bias.size()[0] == weight.size()[0]

def fc_layer_scalar(x_in, weight, bias, device='cpu'):
    """
    Args:
        x_in: [N, C_in] tensor
        weight: [C_out, C_in] tensor
        bias: [C_out] tensor
        device: 'cpu' or 'cuda'
    Returns:
        x_out: [N, C_out] tensor
    """
    assert_fc(x_in, weight, bias)
    # Shapes
    N, C_in = x_in.size()
    C_out, _ = weight.size()
    # Moving to device
    x_in = x_in.to(device)
    weight = weight.to(device)
    bias = bias.to(device)
    # Fully connected
    x_out = torch.zeros([N, C_out]).to(device)
    for c in range(C_out):
        x_out[:,c] = (x_in * weight[c]).sum(1) + bias[c]
    return x_out

def fc_layer_vector(x_in, weight, bias, device='cpu'):
    """
    Args:
        x_in: [N, C_in] tensor
        weight: [C_out, C_in] tensor
        bias: [C_out] tensor
        device: 'cpu' or 'cuda'
    Returns:
        x_out: [N, C_out] tensor
    """
    assert_fc(x_in, weight, bias)
    # Shapes
    N, C_in = x_in.size()
    C_out, _ = weight.size()
    # Moving to device
    x_in = x_in.to(device)
    weight = weight.to(device)
    bias = bias.to(device)
    # Fully connected: [N, C_in] @ [C_in, C_out] -> [N, C_out]
    x_out = torch.matmul(x_in, weight.permute(1,0)) + bias
    return x_out
