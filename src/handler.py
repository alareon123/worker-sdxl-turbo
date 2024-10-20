import runpod
from diffusers import AutoPipelineForText2Image, StableDiffusionXLPipeline, AutoencoderKL
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
import torch
import base64
import io
import time
import gdown
import os

# Создание папки для моделей и LoRA, если она не существует
model_folder = "./models"
lora_folder = "./loras"
os.makedirs(model_folder, exist_ok=True)
os.makedirs(lora_folder, exist_ok=True)

# Функция для скачивания файла с Google Диска
def download_from_google_drive(file_id, output_path):
    google_drive_url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(google_drive_url, output_path, quiet=False)

# Скачивание модели и LoRA с Google Диска
def download_model_and_lora():
    # Загрузка основной модели
    model_file_id = '1i3KMLN2vEZypqk3Cnp55jTYQ4roVjFO_'  # ID файла модели на Google Drive
    model_output_path = os.path.join(model_folder, 'model.safetensors')
    if not os.path.exists(model_output_path):
        download_from_google_drive(model_file_id, model_output_path)

    # Загрузка LoRA весов
    lora_file_id = '1maIK2iejDm02U91AAll6Ih5P5V0AUOkz'  # ID файла LoRA на Google Drive
    lora_output_path = os.path.join(lora_folder, 'lora.safetensors')
    if not os.path.exists(lora_output_path):
        download_from_google_drive(lora_file_id, lora_output_path)

    return model_output_path, lora_output_path

# Загрузка модели и LoRA
model_path, lora_path = download_model_and_lora()

# Загрузка основной модели с Hugging Face
try:
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_single_file(model_path, vae=vae, torch_dtype=torch.float16)
    pipe.to("cuda")

    # Настройка LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["attn1", "attn2"],  # Это могут быть любые слои, которые нужно модифицировать
        lora_dropout=0.1,
        bias="none"
    )

    # Применение LoRA к модели
    pipe.unet = get_peft_model(pipe.unet, lora_config)

    # Загрузка весов LoRA
    with open(lora_path, "rb") as f:
        lora_state_dict = torch.load(f)
    set_peft_model_state_dict(pipe.unet, lora_state_dict)
    print("LoRA успешно загружена.")

except RuntimeError as e:
    print(f"Ошибка при загрузке модели: {e}")
    quit()

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']
    prompt = job_input['prompt']

    time_start = time.time()

    # Генерация изображения
    image = pipe(prompt=prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    print(f"Time taken: {time.time() - time_start}")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    return base64.b64encode(image_bytes).decode('utf-8')


runpod.serverless.start({"handler": handler})
