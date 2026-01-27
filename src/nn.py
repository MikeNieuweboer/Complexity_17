"""Contains the NN class that is used as the update rule for neural cellular automata.

AI usage:
---------
Gemini 3.0 was used to generate docstrings for the functionality in this file.
This was done for each function individually using the inline chat and the
following prompt:
> Analyze the specific function and create a consise docstring
Afterwards manual changes were made to the liking of the coder.
"""

import torch
from torch import nn


class NN(nn.Module):
    """Neural network implementing the update rule for cellular automata.

    Applies Sobel perception filters and neural layers to compute state deltas.
    Reference: https://distill.pub/2020/growing-ca/
    """

    def __init__(self, num_channels: int, hidden_layer_size: int) -> None:
        """Initialize the network.

        Args:
        ----
            num_channels: Length of state vector (C).
            hidden_layer_size: Hidden layer width.

        """
        super().__init__()
        self.num_channels = num_channels

        self.hidden_layer = nn.Conv2d(
            in_channels=3 * num_channels,
            out_channels=hidden_layer_size,
            kernel_size=1,
            bias=False,
        )
        self.output_layer = nn.Conv2d(
            in_channels=hidden_layer_size,
            out_channels=num_channels,
            kernel_size=1,
            bias=False,
        )

        # initialize output layer to 0, as it would not make
        # sense for the grid to initially exhibit chaotic behaviour.
        nn.init.zeros_(self.output_layer.weight)

    def load_weights(
        self, hidden_layer_weight: torch.Tensor, output_layer_weight: torch.Tensor
    ) -> None:
        """Load and validates custom weights for the neural network layers.

        Args:
            hidden_layer_weight (torch.Tensor): Tensor containing weights for the
                                                hidden layer.
            output_layer_weight (torch.Tensor): Tensor containing weights for the output
                                                layer.

        Raises:
            ValueError: If the shape of either input tensor does not match the shape
                of the corresponding layer's existing weights.

        """
        if hidden_layer_weight.shape != self.hidden_layer.weight.shape:
            raise ValueError(  # noqa: TRY003
                "shape mismatch: hidden layer weights should have shape "  # noqa: EM102
                f"{self.hidden_layer.weight.shape}, got {hidden_layer_weight.shape}",
            )

        if output_layer_weight.shape != self.output_layer.weight.shape:
            raise ValueError(  # noqa: TRY003
                "shape mismatch: output layer weights should have shape "  # noqa: EM102
                f"{self.output_layer.weight.shape}, got {output_layer_weight.shape}",
            )

        # wrap in nn.Parameter to let pytorch know they are trainable parameters
        self.hidden_layer.weight = nn.Parameter(hidden_layer_weight)
        self.output_layer.weight = nn.Parameter(output_layer_weight)

    def _to_channels_first(self, tens: torch.Tensor) -> torch.Tensor:
        """Move the channel dimension of a 3D of 4D tensor.

        The channel dimension (back) will be moved to the front.
        This method handles both 3D and 4D tensors:
            - (C, H, W)         -->     (H, W, C)
            - (B, C, H, W)      -->     (B, H, W, C)

        Args:
            tens (torch.Tensor): Input tensor with 3 or 4 dimensions.

        Returns:
            torch.Tensor: The permuted tensor with the channel dimension moved.

        """
        if tens.dim() not in (3, 4):
            raise ValueError(f"Tensor must be 3D or 4D, got shape {tens.shape}")  # noqa: EM102, TRY003

        # (H, W, C) -> (C, H, W)
        if tens.dim() == 3:  # noqa: PLR2004
            return tens.permute(2, 0, 1)

        # (B, H, W, C) -> (B, C, H, W)
        return tens.permute(0, 3, 1, 2)

    def _to_channels_last(self, tens: torch.Tensor) -> torch.Tensor:
        """Move the channel dimension of a 3D of 4D tensor.

        The channel dimension front will be moved to the back.
        This method handles both 3D and 4D tensors:
            - (C, H, W)         -->     (H, W, C)
            - (B, C, H, W)      -->     (B, H, W, C)

        Args:
            tens (torch.Tensor): Input tensor with 3 or 4 dimensions.

        Returns:
            torch.Tensor: The permuted tensor with the channel dimension moved.

        """
        if tens.dim() not in (3, 4):
            raise ValueError(f"Tensor must be 3D or 4D, got shape {tens.shape}")  # noqa: EM102, TRY003

        # (C, H, W) -> (H, W, C)
        if tens.dim() == 3:  # noqa: PLR2004
            return tens.permute(1, 2, 0)

        # (B, C, H, W) -> (B, H, W, C)
        return tens.permute(0, 2, 3, 1)

    def _perceive(self, state_grid: torch.Tensor) -> torch.Tensor:
        """Apply Sobel filters to compute perception vector.

        Args:
            state_grid: Tensor of shape (B, C, H, W).

        Returns:
            Perception tensor of shape (B, 3*C, H, W).
            Concatenates: [Sobel_X, Sobel_Y, Identity]

        """
        _, C, _, _ = state_grid.shape  # noqa: N806

        user_device = state_grid.device
        user_dtype = state_grid.dtype

        # filters
        sobel_x = (
            torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                device=user_device,
                dtype=user_dtype,
            )
            .view(1, 1, 3, 3)
            .repeat(C, 1, 1, 1)
        )

        sobel_y = (
            torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                device=user_device,
                dtype=user_dtype,
            )
            .view(1, 1, 3, 3)
            .repeat(C, 1, 1, 1)
        )

        # create the layer
        conv_x = nn.Conv2d(
            C,
            C,
            3,
            padding=1,
            padding_mode="circular",
            groups=C,
            bias=False,
        )
        conv_y = nn.Conv2d(
            C,
            C,
            3,
            padding=1,
            padding_mode="circular",
            groups=C,
            bias=False,
        )

        # set the weights
        conv_x.weight = nn.Parameter(sobel_x)
        conv_y.weight = nn.Parameter(sobel_y)

        # applying convolution; Conv2D expects (B, C, H, W)
        sobel_x_output = conv_x(state_grid)
        sobel_y_output = conv_y(state_grid)

        # concatenating with identity (B, 3C, H, W)
        perception_grid = torch.cat(
            [sobel_x_output, sobel_y_output, state_grid],
            dim=1,
        )

        return perception_grid  # noqa: RET504

    def forward(self, state_grid: torch.Tensor) -> torch.Tensor:
        """Compute state delta via perception and neural layers.

        Args:
            state_grid: Tensor of shape (H, W, C) or (B, H, W, C).

        Returns:
            Delta tensor of shape (H, W, C) or (B, H, W, C).

        """
        # Move chanel dimension forward
        if state_grid.dim() == 3:
            input_tensor = self._to_channels_first(state_grid.unsqueeze(0))
        else:
            input_tensor = self._to_channels_first(state_grid)

        perception_grid = self._perceive(input_tensor)  # ((B,) 3*C, H, W)
        hidden = torch.relu(self.hidden_layer(perception_grid))
        delta_s = self.output_layer(hidden) # ((B,) C, H, W)

        # revert back to ((B,) H, W, C)
        delta_s = self._to_channels_last(delta_s)

        if state_grid.dim() == 3:  # noqa: PLR2004
            delta_s = delta_s.squeeze(0)

        return delta_s


if __name__ == "__main__":
    pass
