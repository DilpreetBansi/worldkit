"""WorldKit interactive demo for Hugging Face Spaces."""

import gradio as gr

from worldkit import WorldModel
from worldkit.core.config import get_config
from worldkit.core.jepa import JEPA


def create_demo_model():
    """Create a demo model (randomly initialized for the Space)."""
    config = get_config("nano", action_dim=2)
    jepa = JEPA.from_config(config)
    return WorldModel(jepa, config, device="cpu")


model = create_demo_model()


def encode_image(image):
    """Encode an uploaded image to a latent vector."""
    if image is None:
        return "Please upload an image."
    z = model.encode(image)
    return f"Latent vector (dim={len(z)}):\n\n{z[:10].tolist()}...\n\nFull dim: {len(z)}"


def check_plausibility(video_path):
    """Check plausibility of a video."""
    if video_path is None:
        return "Please upload a video."

    import cv2

    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < 30:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        return "Could not read video frames."

    score = model.plausibility(frames)
    status = "NORMAL" if score > 0.3 else "ANOMALY DETECTED"
    return f"Plausibility Score: {score:.3f}\nStatus: {status}\nFrames analyzed: {len(frames)}"


with gr.Blocks(title="WorldKit Demo") as demo:
    gr.Markdown("# WorldKit Demo\n### The open-source world model runtime")
    gr.Markdown("`pip install worldkit` — Train physics-aware AI on a laptop.")

    with gr.Tab("Encode"):
        gr.Markdown("Upload an image to see its latent representation.")
        with gr.Row():
            img_input = gr.Image(type="numpy", label="Input Image")
            text_output = gr.Textbox(label="Latent Vector", lines=5)
        encode_btn = gr.Button("Encode")
        encode_btn.click(encode_image, inputs=img_input, outputs=text_output)

    with gr.Tab("Plausibility"):
        gr.Markdown("Upload a video to check if events are physically plausible.")
        with gr.Row():
            vid_input = gr.Video(label="Input Video")
            plaus_output = gr.Textbox(label="Result", lines=5)
        plaus_btn = gr.Button("Check Plausibility")
        plaus_btn.click(check_plausibility, inputs=vid_input, outputs=plaus_output)

    with gr.Tab("About"):
        gr.Markdown(
            "**WorldKit** is built on the LeWorldModel architecture.\n\n"
            "- 15M parameters\n"
            "- 1 hyperparameter\n"
            "- Single GPU training\n"
            "- 48x faster planning than alternatives\n"
            "- MIT Licensed"
        )

if __name__ == "__main__":
    demo.launch()
