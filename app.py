import streamlit as st
import torch
import os
import cv2
import pandas as pd
from pathlib import Path
import shutil
import tempfile
import subprocess

from utils import (
    load_unet_model,
    predict_interface_mask,
    extract_interface_points,
    draw_interface_overlay,
    extract_frames_from_video,
    plot_sedimentation_curve
)

# ==================== CONFIGURACIÓN DE LA PÁGINA ====================
st.set_page_config(
    page_title="Settling AI",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== INICIALIZACIÓN DE ESTADO ====================
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'csv_path' not in st.session_state:
    st.session_state.csv_path = None
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None


# ==================== FUNCIONES AUXILIARES ====================

def reset_app():
    """Resetear toda la aplicación y limpiar archivos temporales"""
    if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
        try:
            shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
        except:
            pass

    # Resetear estado
    st.session_state.processed = False
    st.session_state.video_path = None
    st.session_state.csv_path = None
    st.session_state.temp_dir = None

    st.rerun()


def process_video(uploaded_file, ensayo_name, time_interval, model, device):
    """Procesar video completo con barra de progreso"""

    # Crear directorio temporal
    temp_dir = tempfile.mkdtemp()
    st.session_state.temp_dir = temp_dir

    # Guardar video subido
    video_input_path = os.path.join(temp_dir, "input_video.mp4")
    with open(video_input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Directorios de trabajo
    frames_dir = os.path.join(temp_dir, "frames")
    processed_dir = os.path.join(temp_dir, "processed")
    Path(frames_dir).mkdir(parents=True, exist_ok=True)
    Path(processed_dir).mkdir(parents=True, exist_ok=True)

    # Paso 1: Extraer frames
    st.info("📹 Extrayendo frames del video...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    frames_list, total_frames = extract_frames_from_video(
        video_input_path,
        frames_dir,
        time_interval
    )

    st.success(f"✓ {total_frames} frames extraídos")

    # Paso 2: Procesar cada frame
    st.info("🤖 Aplicando modelo U-Net...")
    results = []

    for i, frame_path in enumerate(frames_list):
        # Actualizar progreso
        progress = (i + 1) / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Procesando frame {i + 1}/{total_frames} ({progress * 100:.1f}%)")

        # Cargar frame
        image = cv2.imread(frame_path)

        # Predecir interfaz
        mask_binary = predict_interface_mask(model, image, device)
        interface_points = extract_interface_points(mask_binary)

        # Dibujar overlay
        img_result, height_px = draw_interface_overlay(image, interface_points, show_text=True)

        # Guardar frame procesado
        frame_name = os.path.basename(frame_path)
        output_path = os.path.join(processed_dir, frame_name)
        cv2.imwrite(output_path, img_result)

        # Guardar datos
        time_seconds = i * time_interval
        results.append({
            'frame_number': i,
            'time_seconds': time_seconds,
            'time_minutes': time_seconds / 60,
            'height_pixels': height_px if height_px else 0
        })

    progress_bar.progress(1.0)
    status_text.text(f"✓ {total_frames} frames procesados")

    # Paso 3: Crear video con FFmpeg
    st.info("🎬 Generando video final con FFmpeg...")

    output_video_path = os.path.join(temp_dir, f"{ensayo_name}_analysis.mp4")

    # Detectar FFmpeg - portable para local y cloud
    ffmpeg_path = "ffmpeg"  # Por defecto en PATH

    # Si está en Windows local y existe instalación específica
    if os.name == 'nt' and os.path.exists(r"C:\ffmpeg\bin\ffmpeg.exe"):
        ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"

    try:
        # Crear archivo de lista de frames para FFmpeg
        frames_list_file = os.path.join(temp_dir, "frames_list.txt")
        with open(frames_list_file, 'w') as f:
            for frame_name in sorted(os.listdir(processed_dir)):
                frame_path = os.path.join(processed_dir, frame_name)
                # Normalizar rutas para FFmpeg (usar siempre /)
                frame_path_ffmpeg = frame_path.replace('\\', '/')
                f.write(f"file '{frame_path_ffmpeg}'\n")
                f.write(f"duration {1 / 30}\n")

        # Ejecutar FFmpeg
        result = subprocess.run([
            ffmpeg_path,
            '-f', 'concat',
            '-safe', '0',
            '-i', frames_list_file,
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'fast',
            '-crf', '23',
            '-y',
            output_video_path
        ], check=True, capture_output=True, text=True)

        st.success("✓ Video generado con FFmpeg")

    except FileNotFoundError:
        st.error("❌ FFmpeg no está instalado o no se encuentra.")
        st.error(f"Buscando en: {ffmpeg_path}")
        raise
    except subprocess.CalledProcessError as e:
        st.error(f"❌ Error al crear video con FFmpeg")
        st.error(f"Detalles: {e.stderr}")
        raise

    # Paso 4: Guardar CSV
    st.info("💾 Guardando datos...")

    df = pd.DataFrame(results)
    csv_path = os.path.join(temp_dir, f"{ensayo_name}_data.csv")
    df.to_csv(csv_path, index=False)

    st.success("✓ Datos guardados")

    return output_video_path, csv_path, df


# ==================== CARGAR MODELO (solo una vez) ====================
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Descargar modelo desde Hugging Face
    from huggingface_hub import hf_hub_download

    try:
        with st.spinner("📥 Descargando modelo desde Hugging Face..."):
            model_path = hf_hub_download(
                repo_id="dzjuca/settling-model",
                filename="best_model.pth",
                cache_dir="./model_cache"
            )
        st.success("✓ Modelo cargado desde Hugging Face")
    except Exception as e:
        st.error(f"❌ Error al descargar modelo de Hugging Face: {str(e)}")
        st.error("Verifica que el repositorio sea público y el nombre sea correcto")
        st.stop()

    model = load_unet_model(model_path, device)
    return model, device


# ==================== INTERFAZ PRINCIPAL ====================

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🧪 Settling AI")
    st.markdown("**Automated Sedimentation Interface Detection**")

with col2:
    if st.session_state.processed:
        if st.button("🔄 Analizar Nuevo Ensayo", use_container_width=True):
            reset_app()

st.markdown("---")

# Cargar modelo
model, device = load_model()

# Mostrar info del dispositivo
device_name = "🚀 GPU (CUDA)" if device.type == 'cuda' else "💻 CPU"
st.sidebar.success(f"Dispositivo: {device_name}")

# ==================== SIDEBAR - INPUTS ====================
st.sidebar.header("📋 Configuración")

uploaded_file = st.sidebar.file_uploader(
    "Subir video de sedimentación",
    type=['mp4', 'avi', 'mov'],
    help="Sube un video en formato MP4, AVI o MOV"
)

ensayo_name = st.sidebar.text_input(
    "Nombre del ensayo",
    value="Ensayo1",
    help="Ej: Ensayo1, Calibracion, etc."
)

time_interval = st.sidebar.selectbox(
    "Intervalo entre frames",
    options=[5, 30],
    format_func=lambda x: f"{x} segundos",
    help="Tiempo real entre cada frame capturado"
)

process_button = st.sidebar.button(
    "🚀 Procesar",
    type="primary",
    use_container_width=True,
    disabled=(uploaded_file is None or not ensayo_name)
)

# ==================== PROCESAMIENTO ====================
if process_button and not st.session_state.processed:
    with st.spinner("Procesando video..."):
        try:
            video_path, csv_path, df = process_video(
                uploaded_file,
                ensayo_name,
                time_interval,
                model,
                device
            )

            st.session_state.video_path = video_path
            st.session_state.csv_path = csv_path
            st.session_state.df = df
            st.session_state.processed = True

            st.success("✅ Procesamiento completado!")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error durante el procesamiento: {str(e)}")
            st.exception(e)

# ==================== MOSTRAR RESULTADOS ====================
if st.session_state.processed:
    st.markdown("## 📊 Resultados")

    col1, col2 = st.columns(2)

    # COLUMNA IZQUIERDA - VIDEO
    with col1:
        st.markdown("### 🎥 Video Procesado")

        if os.path.exists(st.session_state.video_path):
            with open(st.session_state.video_path, 'rb') as video_file:
                video_bytes = video_file.read()

            # CSS para controlar tamaño del video
            st.markdown(
                """
                <style>
                .stVideo {
                    max-height: 600px;
                    max-width: 100%;
                }
                .stVideo video {
                    max-height: 600px !important;
                    width: auto !important;
                    margin: 0 auto;
                    display: block;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            st.video(video_bytes)

            st.download_button(
                label="⬇️ Descargar Video",
                data=video_bytes,
                file_name=f"{ensayo_name}_analysis.mp4",
                mime="video/mp4",
                use_container_width=True
            )
        else:
            st.error("Video no encontrado")

    # COLUMNA DERECHA - GRÁFICA
    with col2:
        st.markdown("### 📈 Curva de Sedimentación")

        fig = plot_sedimentation_curve(st.session_state.df)
        st.pyplot(fig)

        csv_data = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Descargar CSV",
            data=csv_data,
            file_name=f"{ensayo_name}_data.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Estadísticas
    st.markdown("---")
    st.markdown("### 📊 Estadísticas")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Frames procesados", len(st.session_state.df))

    with col2:
        st.metric("Duración total", f"{st.session_state.df['time_minutes'].max():.1f} min")

    with col3:
        st.metric("Altura inicial", f"{st.session_state.df['height_pixels'].iloc[0]:.1f} px")

    with col4:
        st.metric("Altura final", f"{st.session_state.df['height_pixels'].iloc[-1]:.1f} px")

else:
    st.info("👈 Sube un video y configura los parámetros en el panel izquierdo para comenzar")

    st.markdown("""
    ### 📝 Instrucciones

    1. **Sube un video** de sedimentación (formato MP4, AVI o MOV)
    2. **Ingresa el nombre** del ensayo
    3. **Selecciona el intervalo** de tiempo entre frames
    4. Haz clic en **Procesar** y espera los resultados

    ### ℹ️ Información

    - El modelo detecta automáticamente la interfaz líquido-sedimento
    - La línea **verde** muestra la interfaz detectada
    - La línea **azul** muestra el promedio de altura
    - Los resultados incluyen video procesado y datos en CSV
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Settling AI © 2025 - Powered by U-Net & Streamlit</div>",
    unsafe_allow_html=True
)