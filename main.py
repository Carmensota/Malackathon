
import cv2
import numpy as np
import matplotlib.pyplot as plt

def analizar_incendio(imagen_path, tipo_composicion='swir'):
    #  Procesa una imagen de satélite para detectar áreas quemadas y focos activos.
    # 1. Cargar la imagen
    imagen_bgr = cv2.imread(imagen_path)
    if imagen_bgr is None:
        raise ValueError(f"No se pudo cargar la imagen en la ruta: {imagen_path}")
    
    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    
    # Convertir a espacio de color HSV y escala de grises para análisis
    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. OBJETIVO BÁSICO: Delimitación del área afectada (zona quemada)
    # Las zonas quemadas suelen tener tonos oscuros/marrones oscuros en SWIR o True Color
    # Aplicamos un umbral adaptativo o por rangos de color para aislar la zona quemada
    if tipo_composicion == 'swir' or tipo_composicion == 'wildfire':
        # En composiciones SWIR/Wildfire, el suelo quemado destaca por contraste
        _, mascara_quemada = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    else:
        # Umbralización basada en canales oscuros para color visible
        _, mascara_quemada = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        
    # Limpiar ruido mediante operaciones morfológicas
    kernel = np.ones((5, 5), np.uint8)
    mascara_quemada = cv2.morphologyEx(mascara_quemada, cv2.MORPH_OPEN, kernel)
    mascara_quemada = cv2.morphologyEx(mascara_quemada, cv2.MORPH_CLOSE, kernel)
    
    # Encontrar contornos del perímetro quemado
    contornos, _ = cv2.findContours(mascara_quemada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 3. OBJETIVO AVANZADO: Detección de focos de fuego activo
    # Los puntos calientes o frentes activos brillan intensamente (alta intensidad en canales rojo/amarillo en Wildfire o SWIR)
    if tipo_composicion == 'wildfire' or tipo_composicion == 'swir':
        # Umbral alto para detectar píxeles muy brillantes (fuego activo / altas temperaturas)
        _, mascara_fuego = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    else:
        # Alternativa usando el canal rojo intensificado
        r_channel = imagen_rgb[:, :, 0]
        _, mascara_fuego = cv2.threshold(r_channel, 240, 255, cv2.THRESH_BINARY)
        
    # Encontrar contornos de los focos activos
    contornos_fuego, _ = cv2.findContours(mascara_fuego, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 4. Representación visual de los resultados sobre la imagen original
    resultado_visual = imagen_rgb.copy()
    
    # Dibujar el perímetro del área quemada en amarillo
    cv2.drawContours(resultado_visual, contornos, -1, (255, 255, 0), 2)
    
    # Dibujar los focos de fuego activo en rojo brillante
    cv2.drawContours(resultado_visual, contornos_fuego, -1, (255, 0, 0), 3)
    
    # 5. Mostrar los resultados gráficamente
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(imagen_rgb)
    axes[0].set_title("Imagen Original")
    axes[0].axis('off')
    
    axes[1].imshow(mascara_quemada, cmap='gray')
    axes[1].set_title("Área Afectada (Básico)")
    axes[1].axis('off')
    
    axes[2].imshow(resultado_visual)
    axes[2].set_title("Perímetro + Focos Activos (Avanzado)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Ejemplo de uso (asegúrate de tener una imagen de prueba en el directorio):
# analizar_incendio('ruta_imagen_niebla_swir.png', tipo_composicion='swir')