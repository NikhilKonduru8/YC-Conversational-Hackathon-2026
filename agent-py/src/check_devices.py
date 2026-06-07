"""Hardware check for the Pi: mic, speaker, camera, and OLED.

    uv run src/check_devices.py

Prints the audio devices (so you can set AUDIO_INPUT_DEVICE / AUDIO_OUTPUT_DEVICE
in .env.local if the USB mic/speaker aren't the defaults), tries to grab a camera
frame, and tries to draw to the OLED. Nothing here needs API keys.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv(".env.local")

from config import load_config  # noqa: E402


def check_audio(cfg) -> None:
    print("\n=== AUDIO DEVICES (sounddevice) ===")
    try:
        import sounddevice as sd

        print(sd.query_devices())
        default_in, default_out = sd.default.device
        print(f"\nDefault input index : {default_in}")
        print(f"Default output index: {default_out}")
        print(
            "If your USB mic/speaker aren't the defaults above, set their index "
            "(or a name substring) as AUDIO_INPUT_DEVICE / AUDIO_OUTPUT_DEVICE."
        )

        # Show + validate what .env.local actually configured (this is what
        # jarvis.py will use — the defaults above are just the OS defaults).
        print("\n--- Configured in .env.local (what Jarvis will use) ---")
        for label, value in (
            ("AUDIO_INPUT_DEVICE", cfg.audio.input_device),
            ("AUDIO_OUTPUT_DEVICE", cfg.audio.output_device),
        ):
            if value is None:
                print(f"  {label} = (unset → system default)")
                continue
            try:
                resolved = sd.query_devices(value)
                print(f"  {label} = {value!r}  ->  resolves to: {resolved['name']}")
            except Exception as exc:
                print(f"  {label} = {value!r}  ->  ERROR: {exc} (won't work as-is)")
    except Exception as exc:
        print(f"  audio check failed: {exc!r}")
        print("  -> on the Pi: sudo apt install libportaudio2")


def check_camera(index: int) -> None:
    print(f"\n=== CAMERA (index {index}) ===")
    try:
        import cv2

        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            print(f"  could not open camera {index}. Try CAMERA_INDEX=1, 2, ...")
            print("  list cameras with: ls /dev/video*")
            return
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            print(f"  OK — captured a {w}x{h} frame.")
        else:
            print("  opened the camera but couldn't read a frame.")
    except Exception as exc:
        print(f"  camera check failed: {exc!r}")


def check_oled(cfg) -> None:
    print("\n=== OLED DISPLAY ===")
    try:
        from display import Display

        display = Display(cfg.oled)
        if display.enabled:
            display.show("Jarvis", "hardware OK")
            print(f"  OK — drew to the OLED at 0x{cfg.oled.i2c_address:02x}.")
        else:
            print("  OLED not detected.")
            print("  - install libs:  uv sync --group pi")
            print("  - enable I2C:    sudo raspi-config (Interface Options > I2C)")
            print("  - confirm wiring: i2cdetect -y 1   (should show 3c)")
            print(
                "  - force-on to see the error: JARVIS_OLED=on uv run src/check_devices.py"
            )
    except Exception as exc:
        print(f"  OLED check failed: {exc!r}")


def main() -> None:
    cfg = load_config()
    check_audio(cfg)
    check_camera(cfg.camera.index)
    check_oled(cfg)
    print("\nDone.")


if __name__ == "__main__":
    main()
