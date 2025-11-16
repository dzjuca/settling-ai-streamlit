import cv2
import os
from pathlib import Path
import numpy as np
import torch
import segmentation_models_pytorch as smp
import pandas as pd
from tqdm import tqdm


def select_and_crop_video(input_video_path, output_video_path, display_scale=0.5):
    """
    Permite seleccionar ROI de un video y guarda versión recortada

    Args:
        input_video_path: ruta al video original
        output_video_path: ruta donde guardar video recortado
        display_scale: escala para mostrar frame (0.5 = 50% del tamaño)

    Returns:
        (x, y, w, h): coordenadas del ROI seleccionado
    """

    print(f"Abriendo video: {input_video_path}")

    # Abrir video
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video")
        return None

    # Obtener propiedades del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video info:")
    print(f"  Resolución: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duración: {total_frames / fps:.2f} segundos")

    # Leer primer frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el primer frame")
        cap.release()
        return None

    # Redimensionar para visualización si es necesario
    display_height, display_width = first_frame.shape[:2]
    if display_scale != 1.0:
        display_width = int(width * display_scale)
        display_height = int(height * display_scale)
        frame_display = cv2.resize(first_frame, (display_width, display_height))
    else:
        frame_display = first_frame.copy()

    print(f"\nSelecciona el ROI del tubo con el mouse")
    print("Presiona ENTER cuando termines, ESC para cancelar")

    # Seleccionar ROI
    roi = cv2.selectROI("Seleccionar ROI - Presiona ENTER cuando termines", frame_display, fromCenter=False)
    cv2.destroyAllWindows()

    if roi[2] == 0 or roi[3] == 0:
        print("Selección cancelada")
        cap.release()
        return None

    # Escalar coordenadas al tamaño original si se redimensionó
    if display_scale != 1.0:
        x = int(roi[0] / display_scale)
        y = int(roi[1] / display_scale)
        w = int(roi[2] / display_scale)
        h = int(roi[3] / display_scale)
    else:
        x, y, w, h = roi

    print(f"\nROI seleccionado (tamaño original):")
    print(f"  x={x}, y={y}, width={w}, height={h}")

    # Mostrar preview del ROI
    preview = first_frame[y:y + h, x:x + w]
    if preview.shape[0] > 800:
        preview_scale = 800 / preview.shape[0]
        preview_w = int(preview.shape[1] * preview_scale)
        preview_h = int(preview.shape[0] * preview_scale)
        preview_display = cv2.resize(preview, (preview_w, preview_h))
    else:
        preview_display = preview

    cv2.imshow("Preview ROI - Presiona cualquier tecla para continuar", preview_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Confirmar
    respuesta = input(f"\n¿Proceder a recortar el video completo? (s/n): ")
    if respuesta.lower() != 's':
        print("Operación cancelada")
        cap.release()
        return None

    # Crear directorio de salida si no existe
    Path(os.path.dirname(output_video_path)).mkdir(parents=True, exist_ok=True)

    # Crear video writer para video recortado
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # Reiniciar captura al inicio
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(f"\nRecortando video...")
    print(f"De: {width}x{height} → A: {w}x{h}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Recortar frame
        cropped = frame[y:y + h, x:x + w]

        # Escribir al video de salida
        out.write(cropped)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  Procesados {frame_count}/{total_frames} frames ({frame_count / total_frames * 100:.1f}%)")

    # Liberar recursos
    cap.release()
    out.release()

    # Mostrar info del archivo resultante
    original_size = os.path.getsize(input_video_path) / (1024 ** 3)  # GB
    cropped_size = os.path.getsize(output_video_path) / (1024 ** 3)  # GB

    print(f"\n✓ Video recortado guardado: {output_video_path}")
    print(f"  Frames procesados: {frame_count}")
    print(f"  Tamaño original: {original_size:.2f} GB")
    print(f"  Tamaño recortado: {cropped_size:.2f} GB")
    print(f"  Reducción: {(1 - cropped_size / original_size) * 100:.1f}%")

    return (x, y, w, h)

# ==================== FUNCIONES DE VIDEO Y FRAMES =====================================================================

def extract_frames_from_video_xx(video_path, output_dir, time_interval_seconds):
    """
    Extrae TODOS los frames de un video y los guarda

    Args:
        video_path: ruta al video
        output_dir: carpeta donde guardar frames
        time_interval_seconds: intervalo real entre capturas (para metadata)

    Returns:
        int: número de frames extraídos
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nExtrayendo frames de: {os.path.basename(video_path)}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Tiempo real total: {total_frames * time_interval_seconds / 60:.1f} minutos")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        real_time_seconds = frame_count * time_interval_seconds
        filename = f"frame_{frame_count:04d}_t{real_time_seconds}s.jpg"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, frame)

        frame_count += 1

    cap.release()
    print(f"✓ {frame_count} frames extraídos en: {output_dir}")

    return frame_count


def extract_frames_from_video(video_path, output_dir, time_interval_seconds):
    """
    Extrae TODOS los frames de un video y los guarda

    Args:
        video_path: ruta al video
        output_dir: carpeta donde guardar frames
        time_interval_seconds: intervalo real entre capturas (para metadata)

    Returns:
        tuple: (frames_list, frame_count) - lista de rutas de frames y número total
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nExtrayendo frames de: {os.path.basename(video_path)}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Tiempo real total: {total_frames * time_interval_seconds / 60:.1f} minutos")

    frame_count = 0
    frames_list = []  # ← AGREGAR ESTA LISTA

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        real_time_seconds = frame_count * time_interval_seconds
        filename = f"frame_{frame_count:04d}_t{real_time_seconds}s.jpg"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, frame)
        frames_list.append(output_path)  # ← AGREGAR A LA LISTA

        frame_count += 1

    cap.release()
    print(f"✓ {frame_count} frames extraídos en: {output_dir}")

    return frames_list, frame_count  # ← RETORNAR AMBOS

# ==================== FUNCIONES DE MODELO =============================================================================

def load_unet_model(model_path, device):
    """Cargar modelo U-Net entrenado"""
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Modelo cargado: {os.path.basename(model_path)}")
    print(f"  Época: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")

    return model


def predict_interface_mask(model, image, device):
    """Predecir máscara de interfaz para una imagen"""
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model(img_tensor)
        output = torch.sigmoid(output)
        mask = output.squeeze().cpu().numpy()

    mask_binary = (mask > 0.5).astype(np.uint8) * 255

    return mask_binary


def extract_interface_points(mask_binary):
    """Extraer puntos (x,y) de la línea de interfaz desde máscara binaria"""
    height, width = mask_binary.shape
    interface_points = []

    for x in range(width):
        column = mask_binary[:, x]
        white_pixels = np.where(column == 255)[0]

        if len(white_pixels) > 0:
            y_interface = white_pixels[0]
            interface_points.append((x, y_interface))

    return interface_points


def draw_interface_overlay(image, interface_points, show_text=True):
    """
    Dibujar overlay de interfaz en imagen
    - Línea verde: interfaz completa
    - Línea azul: promedio horizontal

    Returns:
        (img_result, avg_height): imagen con overlay y altura promedio
    """
    img_result = image.copy()

    if len(interface_points) == 0:
        return img_result, None

    # Línea verde (interfaz completa)
    pts = np.array(interface_points, dtype=np.int32)
    cv2.polylines(img_result, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

    # Calcular altura promedio
    avg_y = int(np.mean([p[1] for p in interface_points]))
    height_from_bottom = image.shape[0] - avg_y

    # Línea azul horizontal (promedio)
    cv2.line(img_result, (0, avg_y), (image.shape[1], avg_y), color=(255, 0, 0), thickness=2)

    # Texto opcional
    if show_text:
        text = f"Altura: {height_from_bottom:.1f} px"
        cv2.putText(img_result, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

    return img_result, height_from_bottom


# ==================== PROCESAMIENTO COMPLETO ====================

def process_video_complete(model, frames_dir, output_frames_dir, output_video_path,
                           output_csv_path, time_interval, fps=30, device='cuda'):
    """
    Procesamiento completo: frames → detección → video + CSV

    Args:
        model: modelo U-Net cargado
        frames_dir: carpeta con frames originales
        output_frames_dir: carpeta para guardar frames procesados
        output_video_path: ruta del video final
        output_csv_path: ruta del CSV de resultados
        time_interval: segundos entre frames originales
        fps: fps del video de salida
        device: 'cuda' o 'cpu'

    Returns:
        DataFrame con resultados
    """

    Path(output_frames_dir).mkdir(parents=True, exist_ok=True)

    # Obtener frames
    all_frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])

    print(f"\n{'=' * 60}")
    print(f"PROCESANDO {len(all_frames)} FRAMES")
    print(f"{'=' * 60}\n")

    results = []

    # Procesar cada frame
    for i, frame_name in enumerate(tqdm(all_frames, desc="Procesando frames")):
        frame_path = os.path.join(frames_dir, frame_name)
        image = cv2.imread(frame_path)

        # Predecir interfaz
        mask_binary = predict_interface_mask(model, image, device)
        interface_points = extract_interface_points(mask_binary)

        # Dibujar overlay
        img_result, height_px = draw_interface_overlay(image, interface_points)

        # Guardar frame procesado
        output_path = os.path.join(output_frames_dir, frame_name)
        cv2.imwrite(output_path, img_result)

        # Guardar resultados
        time_seconds = i * time_interval
        results.append({
            'frame_number': i,
            'frame_name': frame_name,
            'time_seconds': time_seconds,
            'time_minutes': time_seconds / 60,
            'height_pixels': height_px,
        })

    # Crear DataFrame
    df = pd.DataFrame(results)

    # Guardar CSV
    df.to_csv(output_csv_path, index=False)
    print(f"\n✓ CSV guardado: {output_csv_path}")

    # Crear video
    print(f"\nGenerando video...")
    frames = sorted([f for f in os.listdir(output_frames_dir) if f.endswith('.jpg')])
    first_frame = cv2.imread(os.path.join(output_frames_dir, frames[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_name in tqdm(frames, desc="Creando video"):
        frame = cv2.imread(os.path.join(output_frames_dir, frame_name))
        video.write(frame)

    video.release()

    video_size = os.path.getsize(output_video_path) / (1024 ** 2)
    print(f"✓ Video guardado: {output_video_path} ({video_size:.1f} MB)")

    return df


def plot_sedimentation_curve_xx(csv_path, output_plot_path=None):
    """Generar gráfica de curva de sedimentación desde CSV"""
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df['time_minutes'], df['height_pixels'], 'b-', linewidth=2, label='Altura interfaz')
    ax.scatter(df['time_minutes'], df['height_pixels'], c='red', s=20, alpha=0.5)

    ax.set_xlabel('Tiempo (minutos)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Altura interfaz (píxeles)', fontsize=12, fontweight='bold')
    ax.set_title('Curva de Sedimentación', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if output_plot_path:
        plt.savefig(output_plot_path, dpi=150)
        print(f"✓ Gráfica guardada: {output_plot_path}")

    plt.show()

    return fig


def plot_sedimentation_curve(data, output_plot_path=None):
    """
    Generar gráfica de curva de sedimentación

    Args:
        data: DataFrame de pandas o ruta a archivo CSV
        output_plot_path: opcional, ruta para guardar la imagen
    """
    import matplotlib.pyplot as plt

    # Si es string, cargar CSV; si es DataFrame, usar directamente
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df['time_minutes'], df['height_pixels'], 'b-', linewidth=2, label='Altura interfaz')
    ax.scatter(df['time_minutes'], df['height_pixels'], c='red', s=20, alpha=0.5)

    ax.set_xlabel('Tiempo (minutos)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Altura interfaz (píxeles)', fontsize=12, fontweight='bold')
    ax.set_title('Curva de Sedimentación', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if output_plot_path:
        plt.savefig(output_plot_path, dpi=150)
        print(f"✓ Gráfica guardada: {output_plot_path}")

    return fig