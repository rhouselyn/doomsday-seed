"""Pinned MediaPipe wheel entry point, called in a child process with local files."""
import json
import sys


if __name__ == "__main__":
    runtime, task, model, image = sys.argv[1:]
    sys.path.insert(0, runtime)
    import mediapipe as mp
    import numpy as np
    from PIL import Image, ImageOps

    vision = mp.tasks.vision
    base = mp.tasks.BaseOptions(model_asset_path=model)
    if task == "pose":
        detector, options = vision.PoseLandmarker, vision.PoseLandmarkerOptions(base_options=base)
    else:
        detector, options = vision.HandLandmarker, vision.HandLandmarkerOptions(base_options=base, num_hands=2)
    with Image.open(image) as source:
        if source.format not in ("PNG", "JPEG", "WEBP") or source.width * source.height > 16_000_000:
            raise ValueError("image exceeds supported format or size")
        rgb = np.array(ImageOps.exif_transpose(source).convert("RGB"))
    with detector.create_from_options(options) as engine:
        found = engine.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    groups = found.pose_landmarks if task == "pose" else found.hand_landmarks
    print(json.dumps({"landmarks": [
        [{"x": point.x, "y": point.y, "z": point.z} for point in landmarks]
        for landmarks in groups
    ]}, allow_nan=False))
