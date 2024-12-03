from ultralytics import YOLO
import os

# Load a model
model = YOLO("yolo11x.pt")  # pretrained YOLO11n model

# Path do diretório de imagens
image_dir = "images"

# Listar todas as imagens do diretório
image_paths = [
    os.path.join(image_dir, file)
    for file in os.listdir(image_dir)
    if file.lower().endswith((".png", ".jpg", ".jpeg"))
]

results = model(image_paths, stream=False)  # retorna um gerador de objetos Results

# Process results generator
for idx, result in enumerate(results):
    # Nome do arquivo para salvar
    output_filename = f"results/detection/result_{idx + 1}.png"  # Usa o índice para gerar nomes únicos
    result.save(filename=output_filename)  # Salva o resultado no disco

    print(f"Resultado salvo em: {output_filename}")
