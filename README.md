# ChameleonNet: Evasión de Identidad Biométrica “A Plena Vista”

> “Si cambiamos la esencia matemática de una imagen, pero mantenemos su forma visual, ¿sigue siendo la misma persona?”

---

## 1. Introducción

En la era de la vigilancia digital omnipresente, la identidad biométrica se ha convertido en un activo transaccional. Los sistemas de reconocimiento facial a gran escala permiten el rastreo masivo de individuos y plantean un desafío directo a la privacidad.

Las soluciones tradicionales de anonimización suelen caer en un dilema binario:

- **Privacidad destructiva**: pixelado, desenfoque u ofuscación extrema que destruyen la utilidad visual de la imagen.
- **Exposición total**: subir imágenes sin modificar, quedando expuesto al rastreo biométrico.

**ChameleonNet** propone una tercera vía.

Es una arquitectura generativa residual diseñada para crear una “capa de invisibilidad digital”: el modelo aprende una máscara de ruido imperceptible que, al sumarse a una imagen facial, **envenena las características semánticas** utilizadas por redes de reconocimiento facial (por ejemplo, FaceNet), haciendo que la identidad del sujeto sea irreconocible o se clasifique como “desconocida”, sin degradar perceptiblemente la calidad visual para el observador humano.

---

## 2. Arquitectura y Metodología

El núcleo de ChameleonNet es un generador que aprende a engañar a un extractor de características biométricas fijo mediante una **“guerra de tres frentes”** durante el entrenamiento.

### 2.1 Componentes principales

| Componente            | Descripción |
|-----------------------|------------|
| **Generador (G)**    | Red tipo U-Net residual. La estructura encoder–decoder con *skip connections* permite generar perturbaciones de alta frecuencia mientras preserva detalles espaciales finos. No sintetiza un rostro nuevo: solo produce el ruido, aplicando una operación del tipo `x_adv = x + G(x)`. |
| **Juez (F)**         | Modelo InceptionResnetV1 (FaceNet) preentrenado en VGGFace2 con pesos congelados. Actúa como adversario estático, proporcionando embeddings biométricos y mapas de características que el generador debe sabotear. |
| **Función de pérdida híbrida** | Combina tres términos competitivos para equilibrar fidelidad visual y evasión biométrica. |

### 2.2 Función de pérdida

La función de pérdida total es:

$$
\mathcal{L}_{\text{total}} =
\lambda_{\text{rec}}\mathcal{L}_{\text{rec}}
+ \lambda_{\text{id}}\mathcal{L}_{\text{id}}
+ \lambda_{\text{attn}}\mathcal{L}_{\text{attn}}.
$$

- **$\mathcal{L}_{rec}$** – *L1 Reconstruction Loss*  
  Fuerza máxima fidelidad visual entre la imagen original \(x\) y la imagen camuflada \(x_{adv}\). Penaliza desviaciones de píxel, manteniendo bordes y detalles de alta frecuencia.

- **$\mathcal{L}_{id}$** – *Cosine Identity Loss*  
  Minimiza la similitud de coseno entre los embeddings biométricos \(F(x)\) y \(F(x_{adv})\), empujándolos hacia la ortogonalidad. En la práctica, esto desplaza la imagen camuflada fuera de su clúster de identidad en el espacio latente.

- **$\mathcal{L}_{attn}$** – *Attention Disruption Loss*  
  Pérdida novedosa que maximiza la divergencia entre mapas de atención intermedios de \(F\). En lugar de preservar la semántica (como en pérdidas perceptuales clásicas), ChameleonNet busca **romper** los patrones de atención, provocando fragmentación de gradiente y dificultando la extracción coherente de rasgos faciales.

---

## 3. Resultados principales (CelebA-HQ 64×64)

En el conjunto de validación de CelebA-HQ (64×64), ChameleonNet logra una disociación efectiva entre apariencia visual e identidad biométrica.

### 3.1 Métricas cuantitativas

| Métrica                          | Objetivo típico                        | Resultado promedio | Estado |
|----------------------------------|----------------------------------------|--------------------|--------|
| **Similitud de identidad (coseno)** | \< 0.4 (desconocido)                  | **-0.0271**        | ✅ Evasión exitosa |
| **PSNR**                         | Alto (\> 25 dB)                        | **27.32 dB**       | ✅ Alta calidad |
| **SSIM**                         | Alto (\> 0.8)                          | **0.8156**         | ✅ Degradación mínima |
| **LPIPS**                        | Bajo (\< 0.05)                         | **0.0450**         | ✅ Diferencias casi imperceptibles |
| **AUC (verificación identidad)** | Cercano a 0.5 para azar                | **0.3853**         | ✅ Por debajo del azar |

- Una similitud de coseno negativa indica que la red biométrica percibe la imagen camuflada como prácticamente opuesta a la identidad original.
- SSIM > 0.8 y LPIPS < 0.05 corresponden a cambios difíciles de detectar por un observador humano.

### 3.2 Análisis forense (XAI)

Mediante Grad-CAM e Integrated Gradients se observa que:

- La atención del modelo de reconocimiento se desplaza desde regiones críticas (ojos, boca) hacia áreas menos informativas o se vuelve difusa.
- Los gradientes se fragmentan: los mapas de relevancia dejan de delinear un rostro coherente y se convierten en patrones ruidosos de alta frecuencia.
- El ruido $\delta$ no es aleatorio; presenta estructura alineada con los rasgos faciales, lo que sugiere que la U-Net aprende un patrón de camuflaje semánticamente informado.


---

## 4. Instalación y uso

Este proyecto está desarrollado en **Python** utilizando **PyTorch**.

### 4.1 Requisitos previos

- Python 3.8+
- PyTorch y torchvision (idealmente con soporte CUDA)
- Librerías adicionales:
  - `numpy`
  - `matplotlib`
  - `Pillow`
  - `scikit-image`
  - `lpips`
  - `facenet_pytorch`
  - (opcional) `jupyter`, `jupyterlab`

### 4.2 Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/chameleonnet.git
cd chameleonnet
```
## 5. Autoría y contacto

- **Autor:** Rodrigo Alfonso Mansilla Dubón  
- **Institución:** Univ
::contentReference[oaicite:0]{index=0}
