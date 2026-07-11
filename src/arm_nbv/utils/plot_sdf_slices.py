import numpy as np
import matplotlib.pyplot as plt


def plot_sdf_slices(
    sdf_grid: np.ndarray, save_path: str, vmin: float = -0.5, vmax: float = 0.5
):
    """
    Helper to plot three-axis slices of an SDF grid and save to file.

    Args:
        sdf_grid: 3D numpy array (nx, ny, nz)
        save_path: Path to save the PNG figure
        vmin, vmax: Colorbar limits for visualization
    """
    sdf_size = sdf_grid.shape[0]
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))

    # Show slices along different axes (edges and interior)
    slice_indices = [0, sdf_size // 4, sdf_size // 2, 3 * sdf_size // 4, sdf_size - 1]

    for i, slice_idx in enumerate(slice_indices):
        # XY slice (Z-axis)
        xy_slice = sdf_grid[:, :, slice_idx]
        axes[0, i].imshow(xy_slice, cmap="RdBu", vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f"XY slice (z={slice_idx})")
        axes[0, i].axis("off")
        # Add SDF values as text annotations (small font)
        for row in range(xy_slice.shape[0]):
            for col in range(xy_slice.shape[1]):
                axes[0, i].text(
                    col,
                    row,
                    f"{xy_slice[row, col]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                )

        # XZ slice (Y-axis)
        xz_slice = sdf_grid[:, slice_idx, :]
        axes[1, i].imshow(xz_slice, cmap="RdBu", vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f"XZ slice (y={slice_idx})")
        axes[1, i].axis("off")
        for row in range(xz_slice.shape[0]):
            for col in range(xz_slice.shape[1]):
                axes[1, i].text(
                    col,
                    row,
                    f"{xz_slice[row, col]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                )

        # YZ slice (X-axis)
        yz_slice = sdf_grid[slice_idx, :, :]
        axes[2, i].imshow(yz_slice, cmap="RdBu", vmin=vmin, vmax=vmax)
        axes[2, i].set_title(f"YZ slice (x={slice_idx})")
        axes[2, i].axis("off")
        for row in range(yz_slice.shape[0]):
            for col in range(yz_slice.shape[1]):
                axes[2, i].text(
                    col,
                    row,
                    f"{yz_slice[row, col]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                )

    plt.suptitle("SDF Visualization (negative=inside, positive=outside)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved SDF visualization to '{save_path}'")
    plt.close()
