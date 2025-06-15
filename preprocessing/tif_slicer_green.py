import os
import cv2
import tifffile as tiff


def preprocess_video(video_name, green_index=1, red_index=0):
    """
    Split a multipage TIFF into separate green‑ and red‑channel image stacks.

    Directory layout created
    ------------------------
    data/processed/<video_name>/
        ├── green_channel/   frame_1.tif … frame_N.tif
        └── red_channel/     frame_1.tif … frame_N.tif   (only for multi‑channel videos)

    For single‑channel (grayscale) videos the frames are **only** saved to
    ``green_channel/`` and a warning is printed. Nothing is written to
    ``red_channel/``.

    Accepted TIFF layouts
    • (N, H, W, C)   – C-channel image per frame
    • (N, C, H, W)   – C single-channel images per frame
    • (N, H, W)      – single-channel grayscale

    Args
    ----
    video_name : str
        Base filename (without ".tif") found under data/raw/.
    green_index : int, default 1
        Index for the green channel.
    red_index   : int, default 0
        Index for the red channel.
    """

    raw_root = os.path.join("data", "raw")
    processed_root = os.path.join("data", "processed", video_name)

    # load TIFF stack
    video_path = os.path.join(raw_root, f"{video_name}.tif")
    video = tiff.imread(video_path)
    print(f"Loaded video {video_path} with shape {video.shape}")

    def save(img, path):
        # ensure uint16 where appropriate
        cv2.imwrite(path, img.astype("uint16") if img.dtype == bool else img)

    j = 0

    # MULTICHANNEL: (N, H, W, C) ---------------------------------------------
    if video.ndim == 4 and video.shape[-1] in (2, 3, 4):
        green_out = os.path.join(processed_root, "green_channel")
        red_out = os.path.join(processed_root, "red_channel")
        os.makedirs(green_out, exist_ok=True)
        os.makedirs(red_out, exist_ok=True)

        for i, frame in enumerate(video, 1):
            if red_index < frame.shape[2]:
                save(frame[..., red_index],
                     os.path.join(red_out, f"frame_{i}.tif"))
            if green_index < frame.shape[2]:
                save(frame[..., green_index],
                     os.path.join(green_out, f"frame_{i}.tif"))
            j+=1

    # MULTICHANNEL: (N, C, H, W) ---------------------------------------------
    elif video.ndim == 4 and video.shape[1] in (2, 3, 4):
        green_out = os.path.join(processed_root, "green_channel")
        red_out = os.path.join(processed_root, "red_channel")
        os.makedirs(green_out, exist_ok=True)
        os.makedirs(red_out, exist_ok=True)

        for i, frame in enumerate(video, 1):
            if red_index < frame.shape[0]:
                save(frame[red_index],
                     os.path.join(red_out, f"frame_{i}.tif"))
            if green_index < frame.shape[0]:
                save(frame[green_index],
                     os.path.join(green_out, f"frame_{i}.tif"))
            j+=1

    # GRAYSCALE: (N, H, W) ----------------------------------------------------
    elif video.ndim == 3:
        print("Single‑channel video detected – saving frames to green_channel only.")
        green_out = os.path.join(processed_root, "green_channel")
        os.makedirs(green_out, exist_ok=True)

        for i, frame in enumerate(video, 1):
            save(frame, os.path.join(green_out, f"frame_{i}.tif"))
            j+=1

    else:
        raise ValueError(f"Unsupported video shape {video.shape}")

    print(f"Finished preprocessing. Processed {j} frames for {video_name}.")
