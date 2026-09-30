# Pre-entrega 3 - Recuperación semántica local

Sistema de recuperación semántica en Python 3.12. Lee documentos locales (markdown, texto y logs), los divide en chunks con metadatos, los guarda con sus embeddings en una base ChromaDB persistente en disco y responde consultas en lenguaje natural, con filtros opcionales por metadatos.

Los embeddings se calculan con **Gemini** (`gemini-embedding-001`) a través del SDK oficial `google-genai`. Por qué así y no con la embedding function de Chroma: ver [la ADR](docs/adr/0001-gemini-embeddings-outside-chroma.md).

## Estructura

```
schemas.py     # Category, EmbeddingSpace, ChunkingSettings, Chunk, SearchQuery, SearchResult
embedder.py    # GeminiEmbedder: task_type asimétrico, batches, reintentos y normalización
indexer.py     # chunking + metadatos, apertura de la colección y CRUD (Indexer)
retriever.py   # búsqueda por similitud coseno con filtros de metadatos
main.py        # demo: indexa data/, dos consultas y un ciclo alta/reindexado/baja
data/          # corpus de prueba de "PayFlow", una plataforma de pagos ficticia
tests/         # pytest con un embedder falso y Chroma en un directorio temporal
```

## Instalación

Con [uv](https://docs.astral.sh/uv/), desde `preentrega-3/`:

```bash
uv sync
```

O con pip:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuración

El `.env` es el mismo de las otras pre-entregas y vive en la raíz del repositorio:

```bash
cp ../.env.example ../.env
```

Completa `GEMINI_API_KEY` con una clave de https://aistudio.google.com/apikey (el plan gratuito alcanza). El resto es opcional:

| Variable | Por defecto | Qué controla |
|---|---|---|
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | modelo de embeddings |
| `CHUNK_SIZE` | `800` | tamaño máximo de cada chunk, en caracteres |
| `CHUNK_OVERLAP` | `120` | caracteres compartidos entre chunks consecutivos (tiene que ser menor que `CHUNK_SIZE`) |
| `CHROMA_DIR` | `chroma_db` | carpeta de la base, relativa a `preentrega-3/` |
| `CHROMA_COLLECTION` | `docs` | nombre de la colección |

## Ejecución

```bash
uv run python main.py
```

La demo:

1. Indexa todo `data/` en `chroma_db/`. Volver a correrla no duplica nada: los IDs de los chunks son determinísticos (`runbook/rollback-de-deploy.md#2`).
2. Hace una consulta directa y la misma consulta filtrada por `category=runbook`.
3. Crea un runbook temporal, lo indexa, lo acorta y lo reindexa (el total de chunks baja), lo busca filtrando por `source` y lo borra.

## Ingesta

- **Categoría**: sale de la carpeta del archivo (`data/arquitectura`, `data/runbook`, `data/log`) y no de la extensión. Un archivo fuera de esas carpetas se rechaza.
- **Chunking**: los `.md` primero se cortan por encabezados con `MarkdownHeaderTextSplitter`, así cada chunk sabe en qué sección está. Después, igual que los `.txt` y `.log`, pasan por `RecursiveCharacterTextSplitter` con `chunk_size` y `chunk_overlap`.
- **Metadatos por chunk**:

  | Campo | Ejemplo |
  |---|---|
  | `source` | `runbook/base-de-datos-no-responde.md` |
  | `category` | `runbook` |
  | `section` | `Mitigación` (vacío si no es markdown) |
  | `chunk_index` | `2` |
  | `ingested_at` | `2026-09-30T15:47:22.728853Z` |

## Embeddings con Gemini

- **Asimétricos**: los chunks se embeben con `task_type=RETRIEVAL_DOCUMENT` y las consultas con `RETRIEVAL_QUERY`. Gemini optimiza cada vector para su rol, y eso mejora la recuperación.
- **768 dimensiones** (`output_dimensionality`). El modelo devuelve 3072 por defecto, pero con un corpus chico no hace falta tanto. Por debajo de 3072 Gemini **no normaliza** los vectores (en la prueba la norma dio 0,58), así que `GeminiEmbedder` los normaliza antes de guardarlos.
- **Espacio de embeddings**: el modelo y la dimensión se guardan en la metadata de la colección. Si después se abre con otro modelo o dimensión, falla con un error claro en vez de mezclar vectores incomparables:

  ```
  La colección 'docs' se creó con gemini-embedding-001 (768 dims) y ahora se pide ... Borra la carpeta de ChromaDB o usa otra colección.
  ```

- **Cuota**: los textos van en lotes de 100 por request. Un 429 o un 5xx se reintenta con backoff exponencial (hasta 5 intentos); cualquier otro error de cliente se propaga sin reintentar.
- **Por qué no `gemini-embedding-2`**: ignora `task_type` (en la prueba devolvió el mismo vector para documento y para consulta), así que se perdería la asimetría.

## Recuperación

```python
SearchQuery(
    text="¿Por qué pagos-api devuelve 503?",
    k=5,                    # 1 a 50
    category="runbook",     # opcional
    source=None,            # opcional, ruta relativa a data/
    min_score=None,         # opcional, entre -1 y 1
)
```

- La colección usa distancia coseno (`hnsw.space = "cosine"`). Cada `SearchResult` trae `score = 1 - distance` (1 es idéntico) y además la `distance` cruda que devuelve Chroma.
- Los filtros se traducen a un `where` de Chroma. Con `category` y `source` juntos se arma un `$and`, así que el chunk tiene que cumplir los dos.
- `min_score` se aplica después de la consulta, sobre los `k` resultados.

## CRUD

```python
indexer.index_directory()                     # alta de todo data/
indexer.index_source(path)                      # alta o reindexado de un archivo
indexer.delete_source("runbook/x.md")         # baja de todos los chunks de un archivo
indexer.count()
```

Reindexar borra los chunks viejos del archivo antes del upsert. Si el archivo quedó más corto, no sobreviven chunks huérfanos de la versión anterior. Los embeddings se piden antes de borrar: si Gemini falla, la versión vieja sigue indexada.

## Salida de ejemplo

Ejecución real contra `gemini-embedding-001`:

```
Indexados 25 chunks de data/ en chroma_db/ (gemini-embedding-001, 768 dims)

== Consulta directa: ¿Por qué pagos-api devuelve 503 cuando se agotan las conexiones a la base? ==
  0.801  [log] log/pagos-api-2026-09-28.log
         2026-09-28 21:14:02 INFO [pagos-api] POST /v1/payments 202 comercio=4312 latency=38ms 2026-09-28 [...]
  0.786  [runbook] runbook/base-de-datos-no-responde.md › Mitigación
         ## Mitigación - Si es saturación: cancelar las consultas largas con `pg_cancel_backend` y bajar [...]
  0.771  [runbook] runbook/base-de-datos-no-responde.md › Síntomas
         # Runbook: la base de datos no responde ## Síntomas - `pagos-api` devuelve 503 y en los logs aparece [...]
  0.756  [arquitectura] arquitectura/capacidad.txt
         Notas de capacidad - revisión de septiembre de 2026 pagos-api corre con 6 réplicas de 1 vCPU. Con 400 [...]
  0.753  [log] log/pagos-api-2026-09-28.log
         2026-09-28 21:14:13 WARN [pagos-worker] mensaje 88213 sin ack, RabbitMQ lo va a reentregar 2026-09-28 [...]

== Consulta filtrada (category=runbook): ¿Por qué pagos-api devuelve 503 cuando se agotan las conexiones a la base? ==
  0.786  [runbook] runbook/base-de-datos-no-responde.md › Mitigación
         ## Mitigación - Si es saturación: cancelar las consultas largas con `pg_cancel_backend` y bajar [...]
  0.771  [runbook] runbook/base-de-datos-no-responde.md › Síntomas
         # Runbook: la base de datos no responde ## Síntomas - `pagos-api` devuelve 503 y en los logs aparece [...]
  0.739  [runbook] runbook/cola-de-pagos-caida.md › Mitigación
         ## Mitigación - Workers caídos: `kubectl rollout restart deployment/pagos-worker`. - Workers bloqueados [...]
  0.725  [runbook] runbook/base-de-datos-no-responde.md › Diagnóstico
         ## Diagnóstico 1. Verificar si el primario está vivo: `kubectl exec -it pgbouncer-0 -- psql -c 'select [...]
  0.699  [runbook] runbook/cola-de-pagos-caida.md › Diagnóstico
         ## Diagnóstico 1. Ver en la consola de RabbitMQ si la cola tiene consumidores conectados. Cero [...]

== CRUD sobre runbook/demo-temporal.md ==
  alta:        2 chunks, total 27
  reindexado:  1 chunks, total 26

== Consulta sobre el runbook actualizado ==
  0.729  [runbook] runbook/demo-temporal.md › Runbook temporal: certificado TLS vencido
         # Runbook temporal: certificado TLS vencido Lo renueva cert-manager solo.

  baja:        1 chunks borrados, total 25
```

La consulta directa mezcla el log del incidente, el runbook y las notas de capacidad. El filtro deja solo los runbooks, que es lo que se busca cuando hay que actuar.

## Tests

```bash
uv run pytest
uv run mypy .
```

Los tests usan un `FakeEmbedder` determinístico (una bolsa de palabras hasheada) y Chroma sobre un directorio temporal, sin tocar la red. `tests/test_gemini.py` sí llama a Gemini y se saltea si no hay `GEMINI_API_KEY`.
