from ultralytics import YOLO

# Load a model
model = YOLO("yolo11l-obb.pt")  # load an official model

image_paths = ["images/image8.png", "images/image9.png", "images/image10.png"]

results = model(image_paths, stream=False)  # retorna um gerador de objetos Results

# Process results generator
for idx, result in enumerate(results):
    output_filename = (
        f"results/obb/result_{idx + 1}.png"  # Usa o índice para gerar nomes únicos
    )
    result.save(filename=output_filename)  # Salva o resultado no disco

    print(f"Resultado salvo em: {output_filename}")
