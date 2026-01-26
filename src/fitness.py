import torch


class FitnessFunctions:
    full_circle_channels = 4

    @staticmethod
    def full_circle(
        grids: torch.Tensor,
        *,
        radius: float = 10.0,
    ) -> torch.Tensor:
        """
        grids: (B, C, H, W) tensor
        returns: (B,) loss tensor, counts mismatches vs target circle mask
        """

        if grids.ndim != 4:
            raise ValueError(f"Expected grids with shape (B, C, H, W), got {tuple(grids.shape)}")

        B, C, H, W = grids.shape

        device = grids.device
        dtype = grids.dtype

        # Coordinates centered in the middle of the grid
        ys = torch.arange(H, device=device, dtype=dtype)
        xs = torch.arange(W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)

        cy = (H - 1) / 2.0
        cx = (W - 1) / 2.0

        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        outside = dist2 >= (radius ** 2)  # (H, W) bool

        # "active" cells for each grid in batch
        active = grids[:, 0:1, :, :] > 0.1  # (B, H, W) bool

        # Your original logic penalized when (outside == active)
        mismatch = active == outside  # (B, H, W) bool

        # Count mismatches per grid
        loss = mismatch.to(torch.float32).sum(dim=(1, 2))  # (B,)
        return loss.detach().cpu().numpy()

