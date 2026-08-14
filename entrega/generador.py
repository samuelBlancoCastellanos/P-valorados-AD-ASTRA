# -*- coding: utf-8 -*-
"""
generador.py — CODEFEST AD ASTRA 2026, Etapa 1.

Lee el archivo de consultas, busca en la base vectorial FAISS y escribe
resultados.jsonl (50 líneas, q001..q050, 3 documentos + 10 fragmentos por
consulta, esquema de la Tabla 2 de la especificación).

Uso:
    python generador.py [ruta_al_archivo_de_consultas] [-o resultados.jsonl]

Si no se pasa ruta, busca un archivo de consultas junto a este script
(consultas*.txt/jsonl/csv/pdf, queries*.*, Extracto_Preguntas*.pdf).

Requisitos: python>=3.9, faiss-cpu, sentence-transformers, numpy
(el índice y la metadata ya vienen construidos en base_vectorial/; este
script NO regenera vectores de pasajes, solo codifica las 50 consultas).

Pipeline de recuperación (sin modelos generativos, §8.3; todas las técnicas
usadas están sancionadas explícitamente por la organización en el documento
de Preguntas Frecuentes: búsqueda híbrida con BM25 y reranking con
cross-encoder, ambas arquitecturas no-decoder):
  1. Codificación de la consulta con el MISMO encoder del índice
     (intfloat/multilingual-e5-base, prefijo "query: ", vector normalizado).
  2. Búsqueda densa exacta (IndexFlatIP, coseno) → pool profundo.
  3. BM25 (implementación propia, determinista) sobre el texto de los
     fragmentos → fusión RRF con el ranking denso.
  4. Deduplicación de fragmentos casi idénticos (hash de texto normalizado).
  5. Reranking opcional con cross-encoder (BAAI/bge-reranker-v2-m3, familia
     XLM-RoBERTa, no genera texto) sobre los primeros RERANK_POOL candidatos.
  6. Fragmentos: top-10 con tope blando por documento.
  7. Documentos: agregación max-pooling con refuerzo por evidencia múltiple
     sobre el pool profundo → top-3 doc_id oficiales (DOC_ID de ADL).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuración (constantes decididas con el arnés de evaluación local;
# ver _audit/EXPERIMENTOS.md del repositorio)
# ---------------------------------------------------------------------------

AQUI = Path(__file__).resolve().parent
DIR_BASE_VECTORIAL = AQUI / "base_vectorial" / "encoder_multilingual-e5-base"

MODELO_ENCODER = "intfloat/multilingual-e5-base"
REVISION_ENCODER = "d128750597153bb5987e10b1c3493a34e5a4502a"  # pin de reproducibilidad

POOL_DENSO = 1000          # profundidad de la búsqueda densa
USAR_BM25 = True           # híbrido sancionado por la FAQ (debe existir la base vectorial: existe)
RRF_K0 = 60
PESO_RRF_DENSO = 1.0
PESO_RRF_BM25 = 0.5        # el denso es la señal principal; BM25 aporta términos exactos

USAR_RERANKER = True
MODELO_RERANKER = "BAAI/bge-reranker-v2-m3"
REVISION_RERANKER = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
RERANK_POOL = 48           # candidatos re-puntuados por el cross-encoder

MAX_FRAGS_POR_DOC = 4      # tope blando por documento en el top-10 de fragmentos
MAX_PALABRAS_SALIDA = 250  # tope duro del reto (§9.2); los chunks vienen ≤240

DOC_TOP_M = 3              # nº de mejores fragmentos por doc para la señal agregada
DOC_PESO_SEGUNDO = 0.30    # peso del refuerzo por evidencia adicional (media de los siguientes m-1)


# ---------------------------------------------------------------------------
# Lectura de consultas (robusta al formato: txt con líneas envueltas, jsonl,
# csv, pdf). Devuelve lista ordenada [(query_id, texto)].
# ---------------------------------------------------------------------------

_QID_RE = re.compile(r"^\s*(q\d{3})\b[.:\-\s]*", re.IGNORECASE)


def _leer_texto(path: Path) -> str:
    datos = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return datos.decode(enc)
        except UnicodeDecodeError:
            continue
    return datos.decode("utf-8", errors="replace")


def _parsear_txt(texto: str) -> "list[tuple[str, str]]":
    consultas = {}
    actual = None
    for linea in texto.splitlines():
        m = _QID_RE.match(linea)
        if m:
            actual = m.group(1).lower()
            consultas[actual] = _QID_RE.sub("", linea, count=1).strip()
        elif actual and linea.strip():
            consultas[actual] += " " + linea.strip()
    return sorted(consultas.items())


def _parsear_jsonl(texto: str) -> "list[tuple[str, str]]":
    consultas = {}
    for linea in texto.splitlines():
        linea = linea.strip()
        if not linea:
            continue
        try:
            obj = json.loads(linea)
        except json.JSONDecodeError:
            return []
        if isinstance(obj, dict):
            qid = str(obj.get("query_id") or obj.get("id") or obj.get("qid") or "").lower()
            q = str(obj.get("query") or obj.get("text") or obj.get("texto")
                    or obj.get("consulta") or obj.get("pregunta") or "")
            if qid and q:
                consultas[qid] = q.strip()
    return sorted(consultas.items())


def _parsear_csv(texto: str) -> "list[tuple[str, str]]":
    import csv as _csv
    import io as _io
    consultas = {}
    for dialecto in (",", ";", "\t"):
        try:
            filas = list(_csv.reader(_io.StringIO(texto), delimiter=dialecto))
        except _csv.Error:
            continue
        for fila in filas:
            if len(fila) >= 2 and re.fullmatch(r"q\d{3}", fila[0].strip().lower()):
                consultas[fila[0].strip().lower()] = fila[1].strip()
        if consultas:
            break
    return sorted(consultas.items())


def _parsear_pdf(path: Path) -> "list[tuple[str, str]]":
    try:
        import fitz  # pymupdf, opcional: solo si las consultas llegan en PDF
    except ImportError:
        print("[ERROR] Las consultas llegaron en PDF y pymupdf no está instalado: pip install pymupdf")
        sys.exit(2)
    doc = fitz.open(str(path))
    texto = "\n".join(p.get_text() for p in doc)
    return _parsear_txt(texto)


def leer_consultas(path: Path) -> "list[tuple[str, str]]":
    if path.suffix.lower() == ".pdf":
        pares = _parsear_pdf(path)
    else:
        texto = _leer_texto(path)
        pares = _parsear_jsonl(texto) or _parsear_csv(texto) or _parsear_txt(texto)
    if not pares:
        print(f"[ERROR] No se pudieron extraer consultas qNNN de {path}")
        sys.exit(2)
    return pares


def encontrar_archivo_consultas() -> "Path | None":
    patrones = ["consultas*", "queries*", "preguntas*", "Extracto_Preguntas*"]
    extensiones = [".txt", ".jsonl", ".json", ".csv", ".pdf"]
    for carpeta in (AQUI, Path.cwd()):
        for pat in patrones:
            for candidato in sorted(carpeta.glob(pat + "*")):
                if candidato.suffix.lower() in extensiones:
                    return candidato
    return None


def validar_consultas(pares) -> None:
    """Aborta con diagnóstico si no hay exactamente q001..q050."""
    esperados = ["q%03d" % i for i in range(1, 51)]
    ids = [q for q, _ in pares]
    faltan = [e for e in esperados if e not in ids]
    sobran = [i for i in ids if i not in esperados]
    if faltan or sobran:
        print("[ERROR] El archivo de consultas no contiene exactamente q001..q050.")
        if faltan:
            print("  Faltan: %s" % ", ".join(faltan))
        if sobran:
            print("  Inesperados: %s" % ", ".join(sobran))
        sys.exit(2)
    vacias = [q for q, t in pares if len(t.strip()) < 5]
    if vacias:
        print("[ERROR] Consultas sin texto (¿fallo de parseo?): %s" % ", ".join(vacias))
        sys.exit(2)


# ---------------------------------------------------------------------------
# BM25 (Okapi) propio: determinista, sin dependencias. Tokenización simple
# sin acentos y en minúsculas, compartida entre indexación y consulta.
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenizar(texto: str) -> "list[str]":
    s = unicodedata.normalize("NFD", texto.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return _TOKEN_RE.findall(s)


class BM25:
    def __init__(self, corpus_tokens, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(corpus_tokens)
        self.doc_len = np.array([len(t) for t in corpus_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        self.tf = []            # list[dict term -> freq]
        df = collections.Counter()
        for toks in corpus_tokens:
            cnt = collections.Counter(toks)
            self.tf.append(cnt)
            for term in cnt:
                df[term] += 1
        self.idf = {t: math.log(1.0 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.postings = collections.defaultdict(list)   # term -> [(doc, freq)]
        for i, cnt in enumerate(self.tf):
            for t, f in cnt.items():
                self.postings[t].append((i, f))

    def buscar(self, consulta: str, k: int) -> "list[tuple[int, float]]":
        puntajes = collections.defaultdict(float)
        for t in _tokenizar(consulta):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, f in self.postings[t]:
                dl = self.doc_len[i]
                puntajes[i] += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return sorted(puntajes.items(), key=lambda x: (-x[1], x[0]))[:k]


# ---------------------------------------------------------------------------
# Recuperación
# ---------------------------------------------------------------------------

def normalizar_texto_dedup(texto: str) -> str:
    return re.sub(r"\W+", "", unicodedata.normalize("NFD", texto.lower()))[:600]


def recortar_a_max_palabras(texto: str, tope: int = MAX_PALABRAS_SALIDA) -> str:
    palabras = texto.split()
    if len(palabras) <= tope:
        return texto
    return " ".join(palabras[:tope])


def main() -> None:
    try:  # consola Windows cp1252: nunca fallar por un print con acentos
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Genera resultados.jsonl desde la base vectorial.")
    parser.add_argument("consultas", nargs="?", type=Path, help="archivo de consultas (txt/jsonl/csv/pdf)")
    parser.add_argument("-o", "--salida", type=Path, default=AQUI / "resultados.jsonl")
    parser.add_argument("--base", type=Path, default=DIR_BASE_VECTORIAL,
                        help="carpeta con index.faiss y metadata.jsonl")
    parser.add_argument("--sin-reranker", action="store_true",
                        help="corrida degradada consciente sin cross-encoder (NO reproduce la entrega)")
    args = parser.parse_args()

    ruta_consultas = args.consultas or encontrar_archivo_consultas()
    if ruta_consultas is None or not Path(ruta_consultas).is_file():
        print("[ERROR] No se encontró archivo de consultas. Se buscó consultas*/queries*/preguntas*"
              " (.txt/.jsonl/.csv/.pdf) en %s y %s. Pásalo como argumento:" % (AQUI, Path.cwd()))
        print("        python generador.py ruta\\al\\archivo_de_consultas.txt")
        sys.exit(2)
    consultas = leer_consultas(Path(ruta_consultas))
    validar_consultas(consultas)
    print(f"Consultas leídas: {len(consultas)} de {ruta_consultas}")

    import faiss
    for req in ("index.faiss", "metadata.jsonl"):
        if not (args.base / req).is_file():
            print(f"[ERROR] Falta {args.base / req}")
            sys.exit(2)
    indice = faiss.read_index(str(args.base / "index.faiss"))
    metadata = []
    with (args.base / "metadata.jsonl").open(encoding="utf-8") as f:
        for linea in f:
            metadata.append(json.loads(linea))
    assert indice.ntotal == len(metadata), "index.faiss y metadata.jsonl desalineados"
    print(f"Índice: {indice.ntotal} vectores, dim {indice.d}")

    # CPU forzada: la corrida del evaluador reproduce byte a byte la nuestra
    # (FAQ: "la ejecución en CPU es suficiente"). MODELO_DIR permite apuntar a
    # una copia local del modelo si no hay red.
    import os
    origen_modelo = os.environ.get("MODELO_DIR", MODELO_ENCODER)
    from sentence_transformers import SentenceTransformer
    try:
        encoder = SentenceTransformer(origen_modelo, revision=REVISION_ENCODER, device="cpu")
    except Exception:
        encoder = SentenceTransformer(origen_modelo, device="cpu")  # caché local con otra revisión

    bm25 = None
    if USAR_BM25:
        print("Construyendo BM25 sobre la metadata (determinista)...", flush=True)
        bm25 = BM25([_tokenizar(m["texto"]) for m in metadata])

    reranker = None
    if USAR_RERANKER and not args.sin_reranker:
        from sentence_transformers import CrossEncoder
        try:
            try:
                reranker = CrossEncoder(MODELO_RERANKER, revision=REVISION_RERANKER, device="cpu")
            except Exception:
                reranker = CrossEncoder(MODELO_RERANKER, device="cpu")  # caché con otra revisión
            print("Cross-encoder de reranking cargado (bge-reranker-v2-m3).")
        except Exception as e:
            # Continuar SIN reranker produciría un resultados.jsonl distinto del
            # entregado (falla silenciosa de reproducibilidad). Mejor fallar claro.
            print(f"[ERROR] No se pudo cargar el reranker {MODELO_RERANKER} ({type(e).__name__}: {e}).")
            print("        Los resultados entregados se generaron CON reranker; sin él no se reproducen.")
            print("        Solución: conexión a HuggingFace o pre-descarga:")
            print("          huggingface-cli download BAAI/bge-reranker-v2-m3")
            print("        (para una corrida degradada consciente: python generador.py --sin-reranker)")
            sys.exit(3)

    vectores_consulta = encoder.encode(
        ["query: " + q for _, q in consultas],
        batch_size=16, normalize_embeddings=True, show_progress_bar=False,
    ).astype(np.float32)

    lineas_salida = []
    for (qid, qtexto), qvec in zip(consultas, vectores_consulta):
        puntajes_densos, ids_densos = indice.search(qvec.reshape(1, -1), POOL_DENSO)
        densos = [(int(i), float(s)) for i, s in zip(ids_densos[0], puntajes_densos[0]) if i >= 0]

        # Fusión RRF denso + BM25 (rangos, robusta a escalas distintas)
        rrf = collections.defaultdict(float)
        denso_score = {}
        for rango, (i, s) in enumerate(densos, start=1):
            rrf[i] += PESO_RRF_DENSO / (RRF_K0 + rango)
            denso_score[i] = s
        if bm25 is not None:
            for rango, (i, _) in enumerate(bm25.buscar(qtexto, POOL_DENSO), start=1):
                rrf[i] += PESO_RRF_BM25 / (RRF_K0 + rango)
        candidatos = sorted(rrf.items(), key=lambda x: (-x[1], metadata[x[0]]["chunk_id"]))

        # Deduplicación por texto normalizado (conserva el mejor rango)
        vistos = set()
        cand_dedup = []
        for i, s in candidatos:
            clave = normalizar_texto_dedup(metadata[i]["texto"])
            if clave in vistos:
                continue
            vistos.add(clave)
            cand_dedup.append((i, s))

        # Reranking opcional del tramo superior
        if reranker is not None:
            tramo = cand_dedup[:RERANK_POOL]
            pares = [(qtexto, metadata[i]["texto"]) for i, _ in tramo]
            puntajes_ce = reranker.predict(pares, batch_size=16, show_progress_bar=False)
            reordenado = sorted(
                zip(tramo, puntajes_ce),
                key=lambda x: (-float(x[1]), metadata[x[0][0]]["chunk_id"]),
            )
            cand_final = [i_s for (i_s, _) in reordenado] + cand_dedup[RERANK_POOL:]
        else:
            cand_final = cand_dedup

        # ---- 10 fragmentos (tope blando por doc) ----
        frags = []
        por_doc = collections.Counter()
        for i, _ in cand_final:
            d = metadata[i]["doc_id"]
            if por_doc[d] >= MAX_FRAGS_POR_DOC:
                continue
            por_doc[d] += 1
            frags.append(i)
            if len(frags) == 10:
                break
        if len(frags) < 10:                      # relleno sin tope (REQ-40: siempre 10)
            for i, _ in cand_final:
                if i not in frags:
                    frags.append(i)
                    if len(frags) == 10:
                        break
        if len(frags) < 10:                      # último recurso: pool denso sin dedup
            for i, _ in densos:
                if i not in frags:
                    frags.append(i)
                    if len(frags) == 10:
                        break
        while len(frags) < 10 and len(frags) < len(metadata):  # índice degenerado
            for i in range(len(metadata)):
                if i not in frags:
                    frags.append(i)
                    break

        # ---- 3 documentos (agregación sobre el pool denso profundo) ----
        por_documento = collections.defaultdict(list)
        for i, s in densos:
            por_documento[metadata[i]["doc_id"]].append(s)
        puntaje_doc = {}
        for d, ss in por_documento.items():
            ss = sorted(ss, reverse=True)[:DOC_TOP_M]
            extra = sum(ss[1:]) / (len(ss) - 1) if len(ss) > 1 else 0.0
            puntaje_doc[d] = ss[0] + DOC_PESO_SEGUNDO * extra
        # refuerzo del reranker: los docs de los 10 fragmentos elegidos suben al frente
        docs_de_frags = []
        for i in frags:
            d = metadata[i]["doc_id"]
            if d not in docs_de_frags:
                docs_de_frags.append(d)
        top_docs = sorted(puntaje_doc.items(), key=lambda x: (-x[1], x[0]))
        slate = []
        for d in docs_de_frags[:2]:              # ancla: hasta 2 docs del top de fragmentos
            if d not in slate:
                slate.append(d)
        for d, _ in top_docs:
            if d not in slate:
                slate.append(d)
            if len(slate) == 3:
                break
        if len(slate) < 3:                       # índice degenerado: pool con <3 docs
            for m in metadata:
                if m["doc_id"] not in slate:
                    slate.append(m["doc_id"])
                if len(slate) == 3:
                    break

        obj = {
            "query_id": qid,
            "documents": [{"rank": r + 1, "doc_id": d} for r, d in enumerate(slate[:3])],
            "fragments": [
                {
                    "rank": r + 1,
                    "chunk_id": metadata[i]["chunk_id"],
                    "doc_id": metadata[i]["doc_id"],
                    "text": recortar_a_max_palabras(metadata[i]["texto"]),
                }
                for r, i in enumerate(frags)
            ],
        }
        lineas_salida.append(json.dumps(obj, ensure_ascii=False))
        print(f"  {qid} listo (top doc: {slate[0] if slate else '-'})", flush=True)

    # Auto-validación (esquema Tabla 2) ANTES de escribir: nunca dejar en
    # disco un resultados.jsonl inválido.
    chunk_ids_meta = {m["chunk_id"] for m in metadata}
    problemas = []
    for n, linea in enumerate(lineas_salida):
        r = json.loads(linea)
        if r["query_id"] != "q%03d" % (n + 1):
            problemas.append("orden: línea %d = %s" % (n + 1, r["query_id"]))
        if len(r["documents"]) != 3 or [d["rank"] for d in r["documents"]] != [1, 2, 3]:
            problemas.append("%s: documents mal formados" % r["query_id"])
        if len(r["fragments"]) != 10 or [fr["rank"] for fr in r["fragments"]] != list(range(1, 11)):
            problemas.append("%s: fragments mal formados" % r["query_id"])
        for fr in r["fragments"]:
            if len(fr["text"].split()) > MAX_PALABRAS_SALIDA:
                problemas.append("%s rank %d: >250 palabras" % (r["query_id"], fr["rank"]))
            if fr["chunk_id"] not in chunk_ids_meta:
                problemas.append("%s rank %d: chunk_id fuera de metadata" % (r["query_id"], fr["rank"]))
    if problemas:
        print("[ERROR] Auto-validación falló; NO se escribió %s:" % args.salida)
        for p in problemas[:20]:
            print("  ! %s" % p)
        sys.exit(1)

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    with args.salida.open("w", encoding="utf-8", newline="\n") as f:
        for linea in lineas_salida:
            f.write(linea + "\n")
    print(f"OK: {args.salida} ({len(lineas_salida)} líneas; auto-validación de esquema superada)")


if __name__ == "__main__":
    main()
