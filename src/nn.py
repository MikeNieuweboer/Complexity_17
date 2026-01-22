import torch
from torch import nn


class NN(nn.Module):
    """Neural network implementing the update rule for cellular automata.

    Applies Sobel perception filters and neural layers to compute state deltas.
    Reference: https://distill.pub/2020/growing-ca/
    """

    def __init__(self, num_channels, hidden_layer_size):
        """Initialize the network.

        Args:
            num_channels: Length of state vector (C).
            hidden_layer_size: Hidden layer width.

        """
        super().__init__()
        self.num_channels = num_channels

        self.hidden_layer = nn.Conv2d(
            in_channels=3 * num_channels, out_channels=hidden_layer_size, kernel_size=1,
        )
        self.output_layer = nn.Conv2d(
            in_channels=hidden_layer_size, out_channels=num_channels, kernel_size=1
        )

        # initialize to 0
        nn.init.zeros_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def perceive(self, state_grid):
        """Apply Sobel filters to compute perception vector.

        Args:
            state_grid: Tensor of shape (H, W, C).

        Returns:
            Perception tensor of shape (1, 3*C, H, W).
            Concatenates: [Sobel_X, Sobel_Y, Identity]

        Doctest
        >>> net = NN(num_channels=3, hidden_layer_size=64)
        >>> state_grid = torch.randn(10, 10, 3) # 10x10 grid with 3 channels
        >>> perception = net.perceive(state_grid)
        >>> perception.shape
        torch.Size([1, 9, 10, 10])

        """
        M = state_grid.shape[2]

        # filters
        sobel_x = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            .view(1, 1, 3, 3)
            .repeat(M, 1, 1, 1)
        )

        sobel_y = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            .view(1, 1, 3, 3)
            .repeat(M, 1, 1, 1)
        )

        # create the layer
        conv_x = nn.Conv2d(
            M, M, 3, padding=1, padding_mode="circular", groups=M, bias=False,
        )
        conv_y = nn.Conv2d(
            M, M, 3, padding=1, padding_mode="circular", groups=M, bias=False,
        )

        # set the weights
        conv_x.weight = nn.Parameter(sobel_x)
        conv_y.weight = nn.Parameter(sobel_y)

        # conv2d expects (B, C, H, W), but state_grid has (H, W, C)
        # transform to (C, H, W)
        input_tensor = state_grid.permute(2, 0, 1)
        # transform to (B, C, H, W)
        input_tensor = input_tensor.unsqueeze(0)

        # applying convolution
        sobel_x_output = conv_x(input_tensor)
        sobel_y_output = conv_y(input_tensor)

        # concatenating with identity
        perception_grid = torch.cat(
            [sobel_x_output, sobel_y_output, input_tensor], dim=1,
        )

        return perception_grid

    def forward(self, state_grid):
        """Compute state delta via perception and neural layers.

        Args:
            state_grid: Tensor of shape (H, W, C).

        Returns:
            Delta tensor of shape (H, W, C).

        """
        perception_grid = self.perceive(state_grid)  # (1, 3*C, H, W)
        hidden = torch.relu(self.hidden_layer(perception_grid))
        delta_s = self.output_layer(hidden)  # (1, C, H, W)

        return delta_s.squeeze(0).permute(1, 2, 0)  # revert to (H, W, C)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
