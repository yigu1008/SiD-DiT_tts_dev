import gradio as gr
import numpy as np
import random

# import spaces #[uncomment to use ZeroGPU]
# from diffusers import SanaPipeline, StableDiffusion3Pipeline, FluxPipeline
from sid import SiDFluxPipeline, SiDSD3Pipeline, SiDSanaPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16

MODEL_OPTIONS = {
    "SiD-Flow-SD3-medium": "YGu1998/SiD-Flow-SD3-medium",
    "SiDA-Flow-SD3-medium": "YGu1998/SiDA-Flow-SD3-medium",
    "SiD-Flow-SD3.5-large": "YGu1998/SiD-Flow-SD3.5-large",
    "SiDA-Flow-SD3.5-large": "YGu1998/SiDA-Flow-SD3.5-large",
    "SiD-Flow-Sana-0.6B-512-res": "YGu1998/SiD-Flow-Sana-0.6B-512-res",
    "SiDA-Flow-Sana-0.6B-512-res": "YGu1998/SiDA-Flow-Sana-0.6B-512-res",
    "SiD-Flow-Sana-1.6B-512-res": "YGu1998/SiD-Flow-Sana-1.6B-512-res",
    "SiD-Flow-Sana-Sprint-0.6B-1024-res": "YGu1998/SiD-Flow-Sana-Sprint-0.6B-1024-res",
    "SiDA-Flow-Sana-Sprint-0.6B-1024-res": "YGu1998/SiDA-Flow-Sana-Sprint-0.6B-1024-res",
    "SiD-Flow-Sana-Sprint-1.6B-1024-res": "YGu1998/SiD-Flow-Sana-Sprint-1.6B-1024-res",
    "SiDA-Flow-Sana-Sprint-1.6B-1024-res": "YGu1998/SiDA-Flow-Sana-Sprint-1.6B-1024-res",
    "SiD-Flow-Flux-1024-res": "YGu1998/SiD-Flow-Flux-1024-res",
    "SiD-Flow-Flux-512-res": "YGu1998/SiD-Flow-Flux-512-res",
}


def load_model(model_choice):
    model_repo_id = MODEL_OPTIONS[model_choice]
    time_scale = 1000.0
    if "Sana" in model_choice:
        pipe = SiDSanaPipeline.from_pretrained(model_repo_id, torch_dtype=torch_dtype)
        if "Sprint" in model_choice:
            time_scale = 1.0
    elif "SD3" in model_choice:
        pipe = SiDSD3Pipeline.from_pretrained(model_repo_id, torch_dtype=torch_dtype)
    elif "Flux" in model_choice:
        pipe = SiDFluxPipeline.from_pretrained(model_repo_id, torch_dtype=torch_dtype)
    else:
        raise ValueError(f"Unknown model type for: {model_choice}")
    pipe = pipe.to(device)
    return pipe, time_scale


MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024


# @spaces.GPU #[uncomment to use ZeroGPU]
def infer(
    prompt,
    seed,
    randomize_seed,
    width,
    height,
    num_inference_steps,
    model_choice,
    progress=gr.Progress(track_tqdm=True),
):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator().manual_seed(seed)

    pipe, time_scale = load_model(model_choice)

    image = pipe(
        prompt=prompt,
        guidance_scale=1,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        generator=generator,
        time_scale=time_scale,
    ).images[0]

    return image, seed


examples = [
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "An astronaut riding a green horse",
    "A delicious ceviche cheesecake slice",
]

css = """
#col-container {
    margin: 0 auto;
    max-width: 640px;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(" # SiD-DiT demo")

        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )

            run_button = gr.Button("Run", scale=0, variant="primary")

        model_choice = gr.Dropdown(
            label="Model Choice",
            choices=list(MODEL_OPTIONS.keys()),
            value="SiD-Flow-SD3-medium",
        )

        result = gr.Image(label="Result", show_label=False)

        with gr.Accordion("Advanced Settings", open=False):

            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,  # Replace with defaults that work for your model
                )

                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,  # Replace with defaults that work for your model
                )

            with gr.Row():
                # guidance_scale = gr.Slider(
                #     label="Guidance scale",
                #     minimum=0.0,
                #     maximum=10.0,
                #     step=0.1,
                #     value=0.0,  # Replace with defaults that work for your model
                # )

                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=4,
                    step=1,
                    value=2,  # Replace with defaults that work for your model
                )

        gr.Examples(examples=examples, inputs=[prompt])
    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[
            prompt,
            seed,
            randomize_seed,
            width,
            height,
            num_inference_steps,
            model_choice,
        ],
        outputs=[result, seed],
    )

if __name__ == "__main__":
    demo.launch()
