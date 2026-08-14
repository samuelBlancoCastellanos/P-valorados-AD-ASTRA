# entrega/ — CODEFEST AD ASTRA 2026 · Etapa 1

Equipo: Los P-Valorados

## Contenido

```
entrega/
  resultados.jsonl        50 líneas (q001..q050), esquema Tabla 2
  generador.py            reproduce resultados.jsonl desde el índice
  informe_tecnico.pdf     informe (≤8 páginas)
  requirements.txt        dependencias exactas del generador
  base_vectorial/
    encoder_multilingual-e5-base/
      index.faiss         IndexFlatIP, 102.736 vectores × 768, normalizados
      metadata.jsonl      102.736 líneas Tabla 1; línea i == id interno FAISS i
```

## Reproducir resultados.jsonl

Probado con Python 3.14 (los pins requieren ≥3.11; el código es compatible ≥3.9):

```bash
python -m venv venv
venv/Scripts/activate            # Windows  (Linux: source venv/bin/activate)
pip install -r requirements.txt
python generador.py ruta/al/archivo_de_consultas.txt
```

- El archivo de consultas puede ser `.txt` (líneas `q001 ...`, admite texto
  envuelto en varias líneas), `.jsonl`, `.csv` o `.pdf` (el PDF requiere
  `pip install pymupdf`). Sin argumento, el script busca `consultas*`/
  `queries*`/`preguntas*` junto a `generador.py` y en el directorio actual.
- Primera ejecución: descarga el encoder `intfloat/multilingual-e5-base`
  (~1,1 GB, revisión pinneada en el código) y el cross-encoder
  `BAAI/bge-reranker-v2-m3` (~2,3 GB) desde HuggingFace. Con caché HF
  presente no hay red. Sin red y sin caché: descargar antes con
  `huggingface-cli download intfloat/multilingual-e5-base` y
  `huggingface-cli download BAAI/bge-reranker-v2-m3`, o apuntar
  `MODELO_DIR` a una copia local del encoder.
- La ejecución es 100 % CPU y determinista (sin aleatoriedad; desempates
  estables por identificador). Duración típica: 10–25 min en CPU moderna
  (la etapa de reranking domina; imprime progreso por consulta). Si el
  reranker no puede cargarse, el generador lo informa y continúa sin él.
- Salida: `resultados.jsonl` junto al script (UTF-8 sin BOM, LF), con
  auto-validación de esquema al final.

## Decisiones de interfaz (aclaraciones oficiales de la FAQ)

- `doc_id` de toda la base y de los resultados = **DOC_ID oficial de ADL**
  (`Indice_Datos_Codefest.xlsx`, hoja "Inventario de Archivos"); `fuente`
  conserva la ruta relativa original del archivo como metadata.
- `chunk_id` de los resultados = el mismo del índice FAISS (metadata.jsonl).
- `formato` = extensión real del archivo de origen en minúsculas.
- Recuperación híbrida (denso + BM25) y reranking con cross-encoder:
  autorizados explícitamente por la organización en el documento de
  Preguntas Frecuentes (ambos son arquitecturas encoder, no decoder, §8.3).
