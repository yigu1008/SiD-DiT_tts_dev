import gradio as gr
import numpy as np
import random

import spaces #[uncomment to use ZeroGPU]
# from diffusers import SanaPipeline, StableDiffusion3Pipeline, FluxPipeline
from sid import SiDFluxPipeline, SiDSD3Pipeline, SiDSanaPipeline
import torch
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16
model_repo_id = "YGu1998/SiD-Flow-Sana-0.6B-512-res"



def load_model(model_repo_id, progress=None):
    if progress is not None:
        progress(0.1, desc=f"Loading {model_choice}...")

    time_scale = 1000.0

    
    pipe = SiDSanaPipeline.from_pretrained(model_repo_id, torch_dtype=torch.bfloat16)
       

    if progress is not None:
        progress(0.5, desc=f"{model_choice} loaded")

    pipe = pipe.to(device)
    return pipe, time_scale


MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024


@spaces.GPU #[uncomment to use ZeroGPU]
def infer(
    prompt,
    seed,
    randomize_seed,
    width,
    height,
    num_inference_steps,
    model_repo_id,
    progress=gr.Progress(track_tqdm=False),
):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator().manual_seed(seed)
    progress(0.0, desc="Preparing model...")
    pipe, time_scale = load_model(model_repo_id)


    progress(0.7, desc="Running inference...")
    image = pipe(
        prompt=prompt,
        guidance_scale=1,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        generator=generator,
        time_scale=time_scale,
    ).images[0]


    progress(1.0, desc="Done")

    pipe.maybe_free_model_hooks()
    del pipe
    torch.cuda.empty_cache()

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
                    minimum=4,
                    maximum=4,
                    step=1,
                    value=4,
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
            model_repo_id,
        ],
        outputs=[result, seed],
    )

if __name__ == "__main__":
    demo.launch()
