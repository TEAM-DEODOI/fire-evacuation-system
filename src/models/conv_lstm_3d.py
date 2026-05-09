"""
3D ConvLSTM for fire spread prediction.

Adapted from ndrplz/ConvLSTM_pytorch (2D version) for our PI-FNO baseline project.

Original repository: https://github.com/ndrplz/ConvLSTM_pytorch
Modifications:
  1. Conv2d -> Conv3d (3D fire grid)
  2. Hidden state extended to 5D: (B, hid, X, Y, Z)
  3. Single-timestep usage wrapper (FireConvLSTM)
  4. Output head: 1x1x1 Conv3d to compress hidden -> 3 channels (T, V, CO)

Input/Output shapes for our project:
  Input:  (B, 5, 60, 40, 6) - [Temperature, Visibility, CO, BuildingMask, TimeEncoding]
  Output: (B, 3, 60, 40, 6) - [Temperature, Visibility, CO] at t+10s
"""

import torch
import torch.nn as nn


class ConvLSTM3DCell(nn.Module):
    """Single 3D ConvLSTM cell. Operates on volumetric data."""

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        """
        Parameters
        ----------
        input_dim : int
            Number of channels in input tensor.
        hidden_dim : int
            Number of channels in hidden state.
        kernel_size : tuple of 3 ints
            Size of the 3D convolutional kernel, e.g. (3, 3, 3).
        bias : bool
            Whether to add bias.
        """
        super().__init__()

        if not (isinstance(kernel_size, tuple) and len(kernel_size) == 3):
            raise ValueError(
                f"kernel_size must be a tuple of 3 ints for 3D ConvLSTM, got {kernel_size}"
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        # 3D padding: same convolution
        self.padding = (
            kernel_size[0] // 2,
            kernel_size[1] // 2,
            kernel_size[2] // 2,
        )
        self.bias = bias

        # Single conv computes all 4 gates simultaneously: i, f, o, g
        self.conv = nn.Conv3d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        """
        Parameters
        ----------
        input_tensor : torch.Tensor
            Shape (B, input_dim, X, Y, Z)
        cur_state : tuple of two torch.Tensor
            (h_cur, c_cur), each of shape (B, hidden_dim, X, Y, Z)

        Returns
        -------
        h_next, c_next : torch.Tensor
            Each of shape (B, hidden_dim, X, Y, Z)
        """
        h_cur, c_cur = cur_state

        # Concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)

        # Split into 4 gates
        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, self.hidden_dim, dim=1
        )

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, volume_size):
        """
        Initialize hidden and cell states with zeros.

        Parameters
        ----------
        batch_size : int
        volume_size : tuple of 3 ints
            (X, Y, Z) spatial dimensions.

        Returns
        -------
        (h, c) : tuple of two torch.Tensor
            Each of shape (batch_size, hidden_dim, X, Y, Z)
        """
        x, y, z = volume_size
        device = self.conv.weight.device
        return (
            torch.zeros(batch_size, self.hidden_dim, x, y, z, device=device),
            torch.zeros(batch_size, self.hidden_dim, x, y, z, device=device),
        )


class ConvLSTM3D(nn.Module):
    """
    Multi-layer 3D ConvLSTM. Processes temporal sequences of 3D volumes.

    Input shape: (B, T, C, X, Y, Z) when batch_first=True
                 (T, B, C, X, Y, Z) when batch_first=False

    For single-timestep usage, set T=1 (see FireConvLSTM wrapper below).
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        num_layers,
        batch_first=True,
        bias=True,
        return_all_layers=False,
    ):
        """
        Parameters
        ----------
        input_dim : int
            Channels in input.
        hidden_dim : int or list of int
            Hidden channels per layer. If int, replicated for all layers.
        kernel_size : tuple of 3 ints, or list of such tuples
            3D kernel size per layer.
        num_layers : int
            Number of stacked ConvLSTM3D layers.
        batch_first : bool
            If True, input is (B, T, ...). If False, (T, B, ...).
        bias : bool
            Whether to use bias in convolutions.
        return_all_layers : bool
            If True, return outputs from all layers. Else only last.
        """
        super().__init__()

        # Normalize kernel_size and hidden_dim to lists of length num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)

        if not (len(kernel_size) == len(hidden_dim) == num_layers):
            raise ValueError("Inconsistent list length for kernel_size/hidden_dim/num_layers")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(
                ConvLSTM3DCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor : torch.Tensor
            Shape (B, T, C, X, Y, Z) if batch_first else (T, B, C, X, Y, Z).
        hidden_state : not implemented (stateful mode disabled)

        Returns
        -------
        layer_output_list : list of torch.Tensor
            Each of shape (B, T, hidden_dim, X, Y, Z).
        last_state_list : list of (h, c) tuples
            Final state at last timestep, per layer.
        """
        if not self.batch_first:
            # (T, B, C, X, Y, Z) -> (B, T, C, X, Y, Z)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4, 5)

        b, seq_len, _, x, y, z = input_tensor.size()

        if hidden_state is not None:
            raise NotImplementedError("Stateful mode not implemented")
        hidden_state = self._init_hidden(batch_size=b, volume_size=(x, y, z))

        layer_output_list = []
        last_state_list = []

        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :, :],
                    cur_state=[h, c],
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, volume_size):
        init_states = []
        for cell in self.cell_list:
            init_states.append(cell.init_hidden(batch_size, volume_size))
        return init_states

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class FireConvLSTM(nn.Module):
    """
    Final fire-prediction model. Wraps ConvLSTM3D with an output head.

    Single-timestep usage: input is one frame, output is the predicted next frame.

    Input shape:  (B, 5, 60, 40, 6)
        Channels: [Temperature, Visibility, CO, BuildingMask, TimeEncoding]
    Output shape: (B, 3, 60, 40, 6)
        Channels: [Temperature, Visibility, CO] at t + dt (dt=10s in our project)

    Example
    -------
    >>> model = FireConvLSTM(in_channels=5, out_channels=3,
    ...                      hidden_dim=32, kernel_size=(3, 3, 3),
    ...                      num_layers=2)
    >>> x = torch.randn(2, 5, 60, 40, 6)
    >>> y = model(x)
    >>> y.shape
    torch.Size([2, 3, 60, 40, 6])
    """

    def __init__(
        self,
        in_channels=5,
        out_channels=3,
        hidden_dim=32,
        kernel_size=(3, 3, 3),
        num_layers=2,
        bias=True,
    ):
        super().__init__()

        # Normalize hidden_dim to list for ConvLSTM3D
        if isinstance(hidden_dim, int):
            hidden_dim_list = [hidden_dim] * num_layers
        else:
            hidden_dim_list = list(hidden_dim)
            if len(hidden_dim_list) != num_layers:
                raise ValueError(
                    f"hidden_dim list length {len(hidden_dim_list)} != num_layers {num_layers}"
                )

        self.convlstm = ConvLSTM3D(
            input_dim=in_channels,
            hidden_dim=hidden_dim_list,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=True,
            bias=bias,
            return_all_layers=False,
        )

        # Output head: hidden_dim of last layer -> out_channels (T, V, CO)
        last_hidden_dim = hidden_dim_list[-1]
        self.output_conv = nn.Conv3d(
            in_channels=last_hidden_dim,
            out_channels=out_channels,
            kernel_size=1,  # 1x1x1 convolution
            bias=True,
        )

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Single-timestep input of shape (B, in_channels, X, Y, Z).

        Returns
        -------
        torch.Tensor
            Predicted next frame of shape (B, out_channels, X, Y, Z).
        """
        # Sanity check
        if x.dim() != 5:
            raise ValueError(
                f"Expected 5D input (B, C, X, Y, Z), got shape {tuple(x.shape)}"
            )

        # Add a singleton time dimension: (B, C, X, Y, Z) -> (B, 1, C, X, Y, Z)
        x = x.unsqueeze(1)

        # Run ConvLSTM3D
        layer_output_list, last_state_list = self.convlstm(x)

        # Take the output of the last layer at the (only) timestep
        # layer_output_list[-1] has shape (B, 1, hidden_dim, X, Y, Z)
        last_layer_output = layer_output_list[-1].squeeze(1)
        # Now shape: (B, hidden_dim, X, Y, Z)

        # Project to output channels via 1x1x1 conv
        out = self.output_conv(last_layer_output)
        # Shape: (B, out_channels, X, Y, Z)

        return out

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =====================================================================
# SANITY CHECK — Run this script directly to verify the module works.
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FireConvLSTM Sanity Check")
    print("=" * 60)

    # Test 1: Basic forward pass on CPU
    print("\n[Test 1] Forward pass on CPU")
    model = FireConvLSTM(
        in_channels=5,
        out_channels=3,
        hidden_dim=32,
        kernel_size=(3, 3, 3),
        num_layers=2,
    )
    print(f"  Total parameters: {model.count_parameters():,}")

    batch_size = 2
    x = torch.randn(batch_size, 5, 60, 40, 6)
    print(f"  Input shape:  {tuple(x.shape)}")

    with torch.no_grad():
        y = model(x)
    print(f"  Output shape: {tuple(y.shape)}")

    expected_shape = (batch_size, 3, 60, 40, 6)
    assert y.shape == expected_shape, f"Expected {expected_shape}, got {tuple(y.shape)}"
    print("  PASS: Output shape correct")

    # Test 2: Backward pass (gradient flow)
    print("\n[Test 2] Backward pass (gradient flow)")
    target = torch.randn(batch_size, 3, 60, 40, 6)
    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()

    has_grad = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
        if p.requires_grad
    )
    assert has_grad, "Some parameters have no gradient"
    print(f"  Loss value: {loss.item():.4f}")
    print("  PASS: All parameters received gradients")

    # Test 3: GPU forward pass (if CUDA available)
    if torch.cuda.is_available():
        print("\n[Test 3] Forward pass on GPU")
        model_gpu = model.cuda()
        x_gpu = x.cuda()
        with torch.no_grad():
            y_gpu = model_gpu(x_gpu)
        print(f"  Output device: {y_gpu.device}")
        print(f"  Output shape:  {tuple(y_gpu.shape)}")

        # Memory usage
        mem_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"  Peak GPU memory (batch={batch_size}): {mem_mb:.1f} MB")
        print("  PASS: GPU forward pass works")
    else:
        print("\n[Test 3] Skipped (no CUDA available)")

    # Test 4: Different batch sizes
    print("\n[Test 4] Variable batch sizes")
    for bs in [1, 4]:
        x_bs = torch.randn(bs, 5, 60, 40, 6)
        with torch.no_grad():
            y_bs = model.cpu()(x_bs)
        assert y_bs.shape == (bs, 3, 60, 40, 6)
        print(f"  Batch size {bs}: {tuple(y_bs.shape)} OK")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
