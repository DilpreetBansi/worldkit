"""Convert video and image data to WorldKit HDF5 format."""

from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


class Converter:
    """Convert various data formats to WorldKit HDF5."""

    def from_video(
        self,
        input_dir: str | Path,
        output: str | Path,
        fps: int = 10,
        action_labels: str | Path | None = None,
        max_frames: int | None = None,
    ) -> Path:
        """Convert MP4/AVI videos to HDF5.

        Args:
            input_dir: Directory of video files.
            output: Output HDF5 path.
            fps: Target FPS (subsamples if needed).
            action_labels: Optional CSV mapping frames to actions.
            max_frames: Max frames per video.

        Returns:
            Path to HDF5 file.
        """
        import cv2

        input_dir = Path(input_dir)
        output = Path(output)
        video_files = sorted(
            list(input_dir.glob("*.mp4"))
            + list(input_dir.glob("*.avi"))
            + list(input_dir.glob("*.mov"))
        )

        if not video_files:
            raise FileNotFoundError(f"No video files found in {input_dir}")

        all_frames = []
        all_actions = []

        for vid_path in tqdm(video_files, desc="Converting videos"):
            cap = cv2.VideoCapture(str(vid_path))
            src_fps = cap.get(cv2.CAP_PROP_FPS)
            skip = max(1, int(src_fps / fps))

            frames = []
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % skip == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    if max_frames and len(frames) >= max_frames:
                        break
                frame_idx += 1
            cap.release()

            if frames:
                all_frames.append(np.stack(frames))
                action_dim = 2
                all_actions.append(
                    np.zeros((len(frames), action_dim), dtype=np.float32)
                )

        max_len = max(f.shape[0] for f in all_frames)
        padded_frames = []
        padded_actions = []

        for frames, actions in zip(all_frames, all_actions):
            T = frames.shape[0]
            if T < max_len:
                pad_f = np.zeros(
                    (max_len - T, *frames.shape[1:]), dtype=frames.dtype
                )
                frames = np.concatenate([frames, pad_f])
                pad_a = np.zeros(
                    (max_len - T, *actions.shape[1:]), dtype=actions.dtype
                )
                actions = np.concatenate([actions, pad_a])
            padded_frames.append(frames)
            padded_actions.append(actions)

        output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output, "w") as f:
            f.create_dataset(
                "pixels", data=np.stack(padded_frames), compression="gzip"
            )
            f.create_dataset(
                "actions", data=np.stack(padded_actions), compression="gzip"
            )

        print(f"WorldKit | Converted {len(video_files)} videos -> {output}")
        return output
