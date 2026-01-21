import gradio as gr
import torch
import random
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from gradio.themes.utils import fonts, sizes
from gradio.themes.base import Base

# --- Custom Gradio Theme (LightTheme) ---
class LightTheme(Base):
    def __init__(self):
        super().__init__(
            primary_hue="indigo",
            secondary_hue="blue",
            neutral_hue="gray",
            font=fonts.GoogleFont("Poppins"),
            radius_size=sizes.radius_md,
            text_size=sizes.text_md
        )

# --- Model Loading and Setup ---
# Determine device (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
# Use float16 for GPU to save VRAM and speed up, float32 for CPU
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Loading models on device: {device} with dtype: {torch_dtype}")

# Load Stable Diffusion v1.4 pipelines
# Using from_pretrained for both text2img and img2img
# Ensure to move to the determined device and use the correct dtype
text2img_pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch_dtype
).to(device)

# Set DPMSolverMultistepScheduler for faster inference
text2img_pipe.scheduler = DPMSolverMultistepScheduler.from_config(text2img_pipe.scheduler.config)

img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch_dtype
).to(device)
img2img_pipe.scheduler = DPMSolverMultistepScheduler.from_config(img2img_pipe.scheduler.config)

# Enable xformers for memory-efficient attention if on CUDA
try:
    if device == "cuda":
        text2img_pipe.enable_xformers_memory_efficient_attention()
        img2img_pipe.enable_xformers_memory_efficient_attention()
        print("xFormers enabled for both pipelines.")
except Exception as e:
    print(f"Could not enable xFormers: {e}. Make sure it's installed (pip install xformers) and compatible with your CUDA setup.")

# Compile UNet for further speedup with PyTorch 2.0+
if device == "cuda" and hasattr(torch, 'compile'):
    try:
        text2img_pipe.unet = torch.compile(text2img_pipe.unet, mode="reduce-overhead", fullgraph=True)
        img2img_pipe.unet = torch.compile(img2img_pipe.unet, mode="reduce-overhead", fullgraph=True)
        print("torch.compile enabled for UNets.")
    except Exception as e:
        print(f"Could not compile UNets: {e}. PyTorch 2.0+ and CUDA required.")

MAX_SEED = 2**32 - 1 # Maximum value for seed

# --- Preset Prompts for Avatar Styles ---
def preset_prompt(type_choice):
    # This dictionary holds all your detailed preset prompts
    presets = {
        # Style-Based
        "Cartoon": "A highly detailed avatar of a teenage male, in cartoon style, with spiky blue hair, wearing a casual hoodie, with a cheerful smile, green eyes, fair skin, freckles on the cheeks, sunny park background, high resolution, sharp focus, cinematic lighting.",
        "Anime": "A highly detailed avatar of a young female, in anime style, with long flowing purple hair, wearing a school uniform, with a playful wink, violet eyes, pale skin, glowing magic aura, cherry blossom background, high resolution, sharp focus, cinematic lighting.",
        "3D": "A highly detailed avatar of a young adult male, in 3D style, with short dark brown hair, wearing a futuristic jacket, with a confident smirk, hazel eyes, medium tan skin, cybernetic eye implant, digital neon city background, high resolution, sharp focus, cinematic lighting.",
        "Pixel": "A highly detailed avatar of a young girl, in pixel art style, with twin buns hairstyle, wearing pixelated armor, with a focused expression, brown eyes, light skin, glowing sword, retro dungeon background, high resolution, sharp focus, cinematic lighting.",
        "Pixar": "A highly detailed avatar of a child female, in Pixar style, with curly black hair, wearing overalls, with a big toothy grin, amber eyes, dark brown skin, backpack full of toys, cozy bedroom background, high resolution, sharp focus, cinematic lighting.",
        "Cyberpunk": "A highly detailed avatar of a teenage male, in cyberpunk style, with slicked neon green hair, wearing a trench coat with circuits, with a fierce gaze, blue cyber eyes, pale skin, face tattoos, rainy city skyline with neon lights, high resolution, sharp focus, cinematic lighting.",
        "Retro": "A highly detailed avatar of a 90s male teen, in retro style, with combed hair, wearing a colorful windbreaker, with a relaxed expression, blue eyes, peach skin, Walkman in hand, arcade background, high resolution, sharp focus, cinematic lighting.",

        # Personality-Based
        "Gamer": "A highly detailed avatar of a young male, in digital gamer style, with messy dark hair, wearing RGB headphones and a hoodie, with an excited expression, dark brown eyes, olive skin, gaming controller in hand, streaming room with LED lights, high resolution, sharp focus, cinematic lighting.",
        "Cool": "A highly detailed avatar of a young adult female, in cool modern style, with sleek silver hair, wearing a leather jacket and shades, with a confident smirk, grey eyes, tan skin, lip piercing, graffiti wall background, high resolution, sharp focus, cinematic lighting.",
        "Nerdy": "A highly detailed avatar of a teenage boy, in nerdy style, with messy brown hair, wearing glasses and a geeky T-shirt, with a curious look, green eyes, fair skin, surrounded by books and gadgets, library background, high resolution, sharp focus, cinematic lighting.",
        "Mysterious": "A highly detailed avatar of an adult male, in mysterious style, with long black hooded hair, wearing a cloak, with a hidden smirk, glowing white eyes, pale skin, mist surrounding him, dark alley background, high resolution, sharp focus, cinematic lighting.",

        # Aesthetic-Based
        "Vaporwave": "A highly detailed avatar of a young male, in vaporwave style, with pastel pink hair, wearing oversized sunglasses and retro clothes, with a chill expression, teal eyes, light skin, grid and palm tree neon backdrop, sunset horizon background, high resolution, sharp focus, cinematic lighting.",
        "Fantasy": "A highly detailed avatar of a young female elf, in fantasy style, with long golden braided hair, wearing elven armor and cloak, with a serene smile, blue eyes, fair skin, glowing staff, enchanted forest background, high resolution, sharp focus, cinematic lighting.",
        "Gothic": "A highly detailed avatar of a gothic female, in gothic style, with black straight hair, wearing corset dress and lace gloves, with a solemn expression, red eyes, pale skin, rose tattoo, dark cathedral background, high resolution, sharp focus, cinematic lighting.",
        "Minimalist": "A highly detailed avatar of an adult male, in minimalist style, with short black hair, wearing a plain white t-shirt, with a calm expression, dark eyes, light beige skin, clean and simple silhouette, white gradient background, high resolution, sharp focus, cinematic lighting.",

        # Cultural-Based
        "Samurai": "A highly detailed avatar of a male samurai, in Japanese traditional style, with tied black hair, wearing a detailed kimono and armor, with a determined stare, brown eyes, tan skin, holding a katana, cherry blossom battlefield background, high resolution, sharp focus, cinematic lighting.",
        "Egyptian": "A highly detailed avatar of an Egyptian queen, in cultural ancient style, with braided hair, wearing a golden headdress and regal robes, with a majestic gaze, dark kohl-lined eyes, bronze skin, Ankh jewelry, pyramid desert background, high resolution, sharp focus, cinematic lighting.",
        "Greek God": "A highly detailed avatar of a male deity, in Greek god style, with curly golden hair, wearing a white toga and golden laurel crown, with a noble look, blue eyes, fair skin, thunderbolt in hand, temple ruins background, high resolution, sharp focus, cinematic lighting.",
        "Indian Prince": "A highly detailed avatar of an Indian prince, in royal Indian style, with styled black hair, wearing a colorful sherwani and turban, with a confident smile, dark brown eyes, brown skin, ornate necklace, palace background, high resolution, sharp focus, cinematic lighting.",
        "African Queen": "A highly detailed avatar of an African queen, in cultural regal style, with braided crown hair, wearing tribal jewelry and colorful robes, with a proud smile, amber eyes, deep brown skin, ceremonial beads, savanna sunset background, high resolution, sharp focus, cinematic lighting.",

        # Gender-Based
        "Male": "A highly detailed avatar of a young male, in realistic style, with short tousled hair, wearing a T-shirt and jacket, with a calm smile, dark brown eyes, tan skin, stubble beard, soft light room background, high resolution, sharp focus, cinematic lighting.",
        "Female": "A highly detailed avatar of a young female, in realistic style, with long wavy hair, wearing a floral dress, with a gentle smile, hazel eyes, fair skin, beauty mark on cheek, cozy room background, high resolution, sharp focus, cinematic lighting.",
    }
    return presets.get(type_choice, "") # Return the prompt or an empty string if not found

# --- Text-to-Image Generation Function ---
def generate_text_to_image(prompt, seed, randomize_seed, guidance_scale, num_inference_steps, progress=gr.Progress()):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device).manual_seed(seed)

    progress(0.5, desc="Generating avatar (Text-to-Image)...")
    image = text2img_pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]
    progress(1.0, desc="Text-to-Image generation complete!")
    return image, seed

# --- Image-to-Image Generation Function ---
def generate_image_to_image(input_image, prompt, strength, guidance_scale, seed, randomize_seed, steps_img, progress=gr.Progress()):
    if input_image is None or prompt.strip() == "":
        raise gr.Error("Upload an image and enter a prompt.")
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    input_image = input_image.resize((512, 512)) # Resize for consistent input resolution
    generator = torch.Generator(device=device).manual_seed(seed)

    progress(0.5, desc="Transforming avatar (Image-to-Image)...")
    output = img2img_pipe(
        prompt=prompt,
        image=input_image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=steps_img,
        generator=generator
    )
    progress(1.0, desc="Image-to-Image transformation complete!")
    return output.images[0], seed

# --- List of all avatar types for dropdowns ---
avatar_styles = [
    # Style-Based
    "Cartoon", "Anime", "3D", "Pixel", "Pixar", "Cyberpunk", "Retro",
    # Personality-Based
    "Gamer", "Cool", "Nerdy", "Mysterious",
    # Aesthetic-Based
    "Vaporwave", "Fantasy", "Gothic", "Minimalist",
    # Cultural-Based
    "Samurai", "Egyptian", "Greek God", "Indian Prince", "African Queen",
    # Gender-Based
    "Male", "Female"
]

# --- Custom CSS for Gradio Interface ---
css = """
/* Animated Gradient Background */
html, body, .gradio-container {
    background: linear-gradient(270deg, #6a11cb, #2575fc, #6a11cb);
    background-size: 600% 600%;
    animation: gradientFlow 15s ease infinite;
    color: white !important;
    color-scheme: only light !important;
    font-family: 'Poppins', sans-serif !important;
}

@keyframes gradientFlow {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Floating panel animation */
.gr-box, .gr-panel, .gr-column, .gr-tabitem {
    background-color: rgba(255, 255, 255, 0.08) !important;
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    animation: floatPanel 6s ease-in-out infinite;
}

@keyframes floatPanel {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-6px); }
    100% { transform: translateY(0px); }
}

/* Button animation + pulse + hover */
.gr-button {
    background: linear-gradient(90deg, #8e2de2, #4a00e0) !important;
    color: white !important;
    font-weight: bold;
    border-radius: 12px;
    padding: 12px 24px;
    border: none;
    transition: all 0.3s ease-in-out;
    animation: pulse 2s infinite;
}
.gr-button:hover {
    transform: scale(1.05) translateY(-2px);
    box-shadow: 0 0 12px rgba(255,255,255,0.3);
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(142, 45, 226, 0.4); }
    70% { box-shadow: 0 0 0 12px rgba(142, 45, 226, 0); }
    100% { box-shadow: 0 0 0 0 rgba(142, 45, 226, 0); }
}

/* Inputs and sliders */
input, textarea, select {
    background-color: rgba(255, 255, 255, 0.1) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 10px;
    padding: 8px;
    transition: background-color 0.3s ease, transform 0.3s ease;
}
input:hover, textarea:hover, select:hover {
    background-color: rgba(255,255,255,0.15) !important;
    transform: scale(1.02);
}

/* Headings */
h1, h2, h3, label {
    text-align: center;
    text-shadow: 0 0 5px rgba(255, 255, 255, 0.3);
    transition: all 0.3s ease-in-out;
}

/* Floating effect for entire Text-to-Image and Image-to-Image sections */
.floating-section {
    animation: floatPanel 6s ease-in-out infinite;
}

/* Button-specific animations */
#text_gen_btn {
    animation: pulseBtn 2s infinite ease-in-out;
}
#img_gen_btn {
    animation: pulseBtnAlt 3s infinite ease-in-out;
}

/* Alternate animation */
@keyframes pulseBtnAlt {
    0% { box-shadow: 0 0 0 0 rgba(255, 100, 180, 0.4); }
    50% { box-shadow: 0 0 20px 8px rgba(255, 100, 180, 0); }
    100% { box-shadow: 0 0 0 0 rgba(255, 100, 180, 0); }
}

/* Animate the Tab Buttons */
.gradio-app .gr-tabs button {
    background: linear-gradient(90deg, #a1c4fd, #c2e9fb);
    color: #333 !important;
    font-weight: bold;
    border-radius: 12px;
    padding: 10px 20px;
    margin: 5px;
    box-shadow: 0 0 15px rgba(0, 255, 255, 0.3);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    animation: floatButton 4s ease-in-out infinite;
}

/* On hover */
.gradio-app .gr-tabs button:hover {
    transform: scale(1.05) translateY(-2px);
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
}

/* Floating animation */
@keyframes floatButton {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-4px); }
    100% { transform: translateY(0px); }
}
"""

# --- Gradio Interface Definition ---
demo = gr.Blocks(css=css, title="Avatar Creator AI", theme=LightTheme())

with demo:
    gr.Markdown("<h2 style='color:#ffd700;text-align:center;'>🌟 Avatar Creator AI</h2>")
    gr.Markdown("<h3 style='text-align:center;'>Create stunning avatars from text or image 🎅</h3>")

    with gr.Tabs():
        with gr.TabItem("Text-to-Image 🎨"):
            with gr.Column(elem_classes=["floating-section"]): # Apply floating effect to this section
                prompt = gr.Textbox(label="Describe your avatar", placeholder="e.g., A highly detailed avatar of a young wizard, magical staff, starry background")
                avatar_type = gr.Dropdown(avatar_styles, label="Or choose an Avatar Style preset")
                # When avatar_type changes, update the prompt textbox with the preset
                avatar_type.change(preset_prompt, inputs=avatar_type, outputs=prompt)
                
                with gr.Row(): # Group seed controls
                    seed_txt = gr.Slider(0, MAX_SEED, step=1, value=0, label="Seed")
                    random_txt = gr.Checkbox(True, label="Randomize Seed")
                
                guidance_txt = gr.Slider(0.0, 10.0, step=0.1, value=7.5, label="Guidance Scale (Higher = more adherence to prompt)")
                steps_txt = gr.Slider(1, 30, step=1, value=20, label="Steps (Lower = Faster, but affects quality)")
                
                txt_btn = gr.Button("🚀 Generate Avatar", elem_id="text_gen_btn")
                txt_img = gr.Image(label="Generated Avatar")
                seed_out_txt = gr.Textbox(label="Used Seed", interactive=False) # Display the actual seed used
                
                txt_btn.click(
                    generate_text_to_image,
                    inputs=[prompt, seed_txt, random_txt, guidance_txt, steps_txt],
                    outputs=[txt_img, seed_out_txt],
                    show_progress=True
                )

        with gr.TabItem("Image-to-Image 🖼"):
            with gr.Column(elem_classes=["floating-section"]): # Apply floating effect to this section
                image_input = gr.Image(type="pil", label="Upload Image (will be resized to 512x512)")
                prompt_img = gr.Textbox(label="Describe transformation (e.g., transform into a superhero, with a futuristic helmet)", info="Describe how you want to change the image.")
                avatar_type_img = gr.Dropdown(avatar_styles, label="Or choose an Avatar Style preset")
                avatar_type_img.change(preset_prompt, inputs=avatar_type_img, outputs=prompt_img)
                
                strength = gr.Slider(0.1, 1.0, step=0.05, value=0.75, label="Transformation Strength (Higher = more change)")
                guidance_img = gr.Slider(1.0, 20.0, step=0.5, value=7.5, label="Guidance Scale")
                steps_img = gr.Slider(1, 50, step=1, value=25, label="Steps (Lower = Faster, but affects quality)")
                
                with gr.Row(): # Group seed controls
                    seed_img = gr.Slider(0, MAX_SEED, step=1, value=0, label="Seed")
                    random_img = gr.Checkbox(True, label="Randomize Seed")
                
                img_btn = gr.Button("💪 Transform", elem_id="img_gen_btn")
                img_result = gr.Image(label="Transformed Avatar")
                seed_out_img = gr.Textbox(label="Used Seed", interactive=False)
                
                img_btn.click(
                    generate_image_to_image,
                    inputs=[image_input, prompt_img, strength, guidance_img, seed_img, random_img, steps_img],
                    outputs=[img_result, seed_out_img],
                    show_progress=True
                )

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()