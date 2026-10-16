import argparse
import os
import sys

import cv2


def count_frames(video_path: str) -> None:
    # Ensure the file exists
    if not os.path.exists(video_path):
        print(f"Error: The file '{video_path}' does not exist.")
        sys.exit(1)

    # Open the video file stream
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: OpenCV could not open the video file '{video_path}'.")
        sys.exit(1)

    # 1. Read the metadata header values
    header_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Analyzing file: {os.path.basename(video_path)}")
    print(f"  - Video FPS (Header): {fps}")
    print(f"  - Frame Count (Header Metadata): {header_count}")
    print("  - Counting actual physical frames (decoding video stream)...")

    # 2. Physically count every frame by reading the file
    actual_count = 0
    while True:
        ret, _frame = cap.read()
        if not ret:
            # ret becomes False when we hit the end of the video file
            break
        actual_count += 1

    # Release the video object back to the system
    cap.release()

    # 3. Print the comparison results
    print("-" * 50)
    print(f"👉 Actual Decoded Frames: {actual_count}")
    print("-" * 50)

    if header_count != actual_count:
        diff = actual_count - header_count
        print(
            f"⚠️ Diagnostic Mismatch: The video actually contains {abs(diff)} "
            f"{'more' if diff > 0 else 'fewer'} frame(s) than the metadata claims."
        )
    else:
        print("✅ Success: The metadata header matches the actual decoded frame count.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count the exact number of physical frames in an MP4 video file by decoding it entirely."
    )
    parser.add_argument("video_path", type=str, help="Path to the .mp4 video file.")
    args = parser.parse_args()

    count_frames(args.video_path)
