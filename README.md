# CODEFEST AD ASTRA 2026 — Etapa 1

**Equipo: Los P-Valorados** — Antonio (ingesta) · Samuel (fragmentación) · Lucas (codificación e
índice) · Pao (recuperación y evaluación)

Base de conocimiento vectorial para el análisis de fenómenos aeroespaciales y territoriales.
Este repositorio contiene **el entregable de la Etapa 1**.

## ⚠️ Antes de clonar: Git LFS

El índice FAISS (301 MB) y su metadata (118 MB) se almacenan con **Git LFS**. Sin LFS instalado, el
clon trae archivos de texto con punteros en lugar de los datos reales:

```bash
git lfs install
git clone https://github.com/samuelBlancoCastellanos/P-valorados-AD-ASTRA.git
```

Si ya clonaste sin LFS: `git lfs install && git lfs pull`.

## Contenido

```
entrega/
  resultados.jsonl        50 líneas (q001..q050), esquema de la Tabla 2
  generador.py            reproduce resultados.jsonl desde el índice
  informe_tecnico.pdf     informe técnico (4 páginas)
  requirements.txt        dependencias exactas del generador
  README.md               instrucciones detalladas de reproducción
  base_vectorial/
    encoder_multilingual-e5-base/
      index.faiss         IndexFlatIP, 102.736 vectores × 768 dim, normalizados
      metadata.jsonl      102.736 líneas (Tabla 1); línea i == id interno FAISS i
```

## Resumen técnico

- **Corpus:** 1.826 documentos oficiales de ADL en siete formatos (PDF, JSON, CSV, XLSX, imágenes,
  TXT y teselas PBF), sobre los tres fenómenos del reto.
- **Fragmentación:** híbrida dirigida por formato (prosa con PySBD a ~256 tokens; fila atómica en
  tabulares; atributos por municipio en PBF) → 102.736 fragmentos.
- **Codificación:** `intfloat/multilingual-e5-base` (MIT, 768 dim, prefijos `passage:`/`query:`,
  *mean pooling*, vectores normalizados).
- **Índice:** `IndexFlatIP` de FAISS — búsqueda exacta equivalente a similitud coseno (§8.2).
- **Recuperación:** densa + señal léxica fusionadas con RRF y reordenamiento con *cross-encoder*
  (arquitecturas encoder, sin modelos generativos, conforme al §8.3).

## Reproducir la entrega

Instrucciones completas (entorno, modelos, tiempos) en [`entrega/README.md`](entrega/README.md):

```bash
cd entrega
python -m venv venv && venv/Scripts/activate    # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
python generador.py ruta/al/archivo_de_consultas.txt
```

El detalle de diseño, decisiones y mediciones está en
[`entrega/informe_tecnico.pdf`](entrega/informe_tecnico.pdf).
