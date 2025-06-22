import os
import tempfile
import io
from tqdm import tqdm
import imageio
import cairosvg
from biked_commons.rendering.rendering import RenderingEngine  # Adjust if needed


def render_to_gif(
    data,
    renderer=None,
    gif_filename=None,
    duration=0.1,
    max_frames=None,
    rider_dims = None
):
    """
    Renders each row of a DataFrame using the given renderer and creates a GIF.

    Args:
        data (pd.DataFrame): The dataframe containing rows to render.
        renderer (RenderingEngine or None): If None, a new RenderingEngine will be created.
        gif_filename (str or None): If provided, saves GIF to this path. Otherwise, only returns it in memory.
        duration (float): Frame duration in seconds.
        max_frames (int or None): If set, limits to the first N frames.
        rider_dims (array-like or None): Optional dimensions for the rider, if applicable.

    Returns:
        BytesIO: In-memory GIF object (can be passed to IPython.display.Image).
    """
    if renderer is None:
        renderer = RenderingEngine(number_rendering_servers=1, server_init_timeout_seconds=120)

    temp_dir = tempfile.mkdtemp()
    png_files = []

    data_iter = data.iterrows()
    if max_frames is not None:
        data_iter = list(data_iter)[:max_frames]

    for i, (_, row) in enumerate(tqdm(data_iter, desc="Rendering frames", total=max_frames or len(data))):
        try:
            res = renderer.render_clip(row, rider_dims=rider_dims)
            svg_bytes = res.image_bytes

            png_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
            cairosvg.svg2png(bytestring=svg_bytes, write_to=png_path)
            png_files.append(png_path)

        except Exception as e:
            print(f"[Warning] Rendering failed for row {i}: {e}")

    # Always write to memory
    gif_buffer = io.BytesIO()
    with imageio.get_writer(gif_buffer, mode="I", format="GIF", duration=duration) as writer:
        for png_file in png_files:
            writer.append_data(imageio.imread(png_file))
    gif_buffer.seek(0)

    # Optionally write to disk
    if gif_filename is not None:
        with open(gif_filename, "wb") as f:
            f.write(gif_buffer.getbuffer())

    return gif_buffer
