# Pre-entrega 1 — Cliente de LLM robusto y asíncrono

Cada pre-entrega es un proyecto independiente, con sus dependencias, tests y README propios. Las dos primeras usan Groq como único proveedor (ver [la ADR](preentrega-1/docs/adr/0001-groq-only-provider.md)); la tercera y la cuarta usan Gemini para los embeddings (ver [la ADR de la P3](preentrega-3/docs/adr/0001-gemini-embeddings-outside-chroma.md) y [la de la P4](preentrega-4/docs/adr/0001-gemini-embeddings-at-1536.md)); la quinta vuelve a Groq para el agente (ver [su ADR](preentrega-5/docs/adr/0001-groq-instead-of-openai.md)).

| Carpeta | Contenido |
|---|---|
| [`preentrega-1/`](preentrega-1/README.md) | Cliente LLM asíncrono unificado: respuesta completa y streaming, validación con Pydantic, reintentos con backoff. |
| [`preentrega-2/`](preentrega-2/README.md) | Pipeline LCEL que convierte logs y descripciones de arquitectura en un análisis técnico validado, con reintentos y autocorrección. |
| [`preentrega-3/`](preentrega-3/README.md) | Recuperación semántica local: chunking con metadatos, ChromaDB persistente con embeddings de Gemini y búsqueda con filtros. |
| [`preentrega-4/`](preentrega-4/README.md) | RAG híbrido en la nube: Pinecone Serverless con namespaces, BM25 + vectorial con EnsembleRetriever y evaluación con Precision@k y Recall@k. |
| [`preentrega-5/`](preentrega-5/README.md) | Agente ReAct con LangGraph: tres herramientas propias, ciclo autónomo con ToolNode y tools_condition, memoria persistente en SQLite por thread_id y traza en JSON. |

## Puesta en marcha

Requiere Python 3.12 y [uv](https://docs.astral.sh/uv/) (o pip con el `requirements.txt` de cada carpeta).

```bash
git clone <url-del-repo>
cd pre-deliverys
cp .env.example .env        # completa GROQ_API_KEY, GEMINI_API_KEY y PINECONE_API_KEY
```

El `.env` de la raíz sirve para todas las pre-entregas. Después, dentro de cada carpeta:

```bash
cd preentrega-2
uv sync
uv run python main.py
uv run pytest
```

# Pre-entrega 2 - Pipeline de extracción validada

Pipeline en Python 3.12 que recibe texto técnico sin estructura (un log de error o una descripción de arquitectura) y lo convierte en un objeto Pydantic validado, usando LangChain Expression Language (LCEL) sobre Groq, con reintentos automáticos y fallos controlados.

> Como en la Pre-entrega 1, el único proveedor es **Groq** (ver `preentrega-1/docs/adr/0001-groq-only-provider.md`), aquí a través de `langchain-groq`.

## Estructura

```
schemas.py     # TechnicalAnalysis (con sus validadores), ExtractionOutcome, ExtractionFailure
prompts.py     # ChatPromptTemplate con input_text y las directivas de contexto
chain.py       # cadena LCEL, reintentos y traducción de errores
main.py        # demo: log, arquitectura, caso ambiguo y API key inválida
tests/         # pytest, con un chat model falso (sin llamadas reales)
```

## Instalación

Con [uv](https://docs.astral.sh/uv/), desde `preentrega-2/`:

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

El `.env` se comparte con la Pre-entrega 1 y vive en la raíz del repositorio; `load_dotenv()` lo encuentra subiendo de carpeta:

```bash
cp ../.env.example ../.env
```

Completa `GROQ_API_KEY` con una clave de https://console.groq.com/keys (el plan gratuito alcanza). `GROQ_MODEL` es opcional; por defecto se usa `openai/gpt-oss-120b`. Sirve cualquier modelo de Groq que soporte tool calling.

## Ejecución

```bash
uv run python main.py      # o: python main.py con el venv activado
```

La demo procesa en paralelo tres textos y después prueba una API key inválida.

## La cadena

```python
structured = llm.with_structured_output(TechnicalAnalysis, include_raw=True)
attempt = (
    RunnableLambda(_add_correction)
    | EXTRACTION_PROMPT
    | _reporting_failed_calls(structured)
    | RunnableLambda(_accept)
).with_retry(retry_if_exception_type=_TRANSIENT_PROVIDER_ERRORS, stop_after_attempt=3, ...)

chain = attempt.with_fallbacks(
    [attempt] * 2,
    exceptions_to_handle=(InvalidOutputError, TruncatedOutputError),
    exception_key="previous_error",
)
```

- **Prompt**: `ChatPromptTemplate` con un mensaje de sistema, otro humano y un `MessagesPlaceholder` opcional para la corrección. Las variables son `input_text`, `criticality_criteria` y `output_language`. Las dos últimas tienen valor por defecto vía `.partial()` y se pueden sobrescribir:

  ```python
  from prompts import EXTRACTION_PROMPT
  EXTRACTION_PROMPT.partial(output_language="inglés")
  ```

- **Modelo**: `ChatGroq` con `temperature=0`, `max_tokens=1024` y `max_retries=0`, para que los únicos reintentos sean los de la cadena.
- **Salida estructurada**: `with_structured_output(..., include_raw=True)` devuelve el `AIMessage` crudo junto al objeto parseado. Así se puede revisar `finish_reason` y el error de validación sin que la cadena explote antes de tiempo.
- **Aceptación** (`_accept`): rechaza la respuesta si `finish_reason == "length"` (salida truncada), si Pydantic no la validó o si el modelo no llamó a la herramienta.
- **Reintento con autocorrección**: con `temperature=0`, repetir la misma petición devuelve la misma respuesta mal formada. Por eso una respuesta rechazada no se reintenta con `with_retry`, sino con `with_fallbacks(exception_key="previous_error")`. Cada nuevo intento recibe el motivo del rechazo (el error de validación, o que se cortó por tokens) como un mensaje extra y puede corregirse. `with_retry` queda para los fallos transitorios de la API, donde repetir sí sirve.

## Esquema

```python
class TechnicalAnalysis(BaseModel):
    tipo_de_entrada: InputKind              # arquitectura | log_error | ambiguo
    tecnologias: list[str]                  # sin espacios sobrantes ni duplicados, al menos una
    componentes_afectados: list[str]        # misma limpieza, puede quedar vacía
    nivel_de_criticidad: CriticalityLevel   # baja | media | alta
    resumen_tecnico: str                    # al menos 20 caracteres
```

Validadores:

- `@field_validator("tecnologias")`: limpia espacios, descarta vacías y valores de relleno (`N/A`, `desconocida`, `ninguna`...), quita duplicados sin distinguir mayúsculas y exige al menos una.
- `@field_validator("componentes_afectados")`: la misma limpieza.
- `@model_validator(mode="after")`: un `log_error` debe indicar al menos un componente afectado, y un texto `ambiguo` no puede tener criticidad `alta`.
- `extra="forbid"`: cualquier campo inventado por el modelo invalida la respuesta.

## Resiliencia

| Situación | Reintenta | Resultado final |
|---|---|---|
| Respuesta que no pasa la validación de Pydantic | sí, con el error como corrección | `invalid_output` |
| El modelo no llama a la herramienta, o Groq responde `tool_use_failed` | sí, con el error como corrección | `invalid_output` |
| `finish_reason == "length"` | sí, pidiendo una respuesta más breve | `truncated_output` |
| Rate limit (429), 5xx, timeout, error de red | sí, la misma petición con backoff | `provider` |
| API key inválida, request mal formada | no | `provider` |

Hay 3 intentos como máximo para cada tipo de fallo. Los fallos transitorios esperan con backoff exponencial y jitter (2 s, 4 s... hasta 20 s). Si se agotan los intentos de corrección, el fallo informa el motivo del primer rechazo, que es el que `with_fallbacks` relanza. Cada intento descartado queda en el log como `WARNING`. `extract()` y `extract_many()` nunca lanzan por estos motivos: devuelven un `ExtractionOutcome` con `analysis` o con `failure`, más la latencia.

## Salida de ejemplo

Ejecución real contra `openai/gpt-oss-120b`.

**Log de error** (pool de PostgreSQL agotado, 503 en cascada):

```json
{
  "tipo_de_entrada": "log_error",
  "tecnologias": ["psycopg_pool", "FastAPI", "uvicorn", "nginx", "celery", "PostgreSQL"],
  "componentes_afectados": ["api-pedidos", "PostgreSQL", "nginx", "worker-facturacion", "celery"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "El pool de conexiones psycopg no pudo obtener una conexión en 30 s, provocando errores 503 en la API FastAPI (uvicorn) y en el upstream de nginx; además una tarea de Celery en worker-facturacion falló por OperationalError de PostgreSQL, indicando una caída del servicio por problemas de base de datos."
}
```

**Descripción de arquitectura** (React + Django + Redis + Celery en Kubernetes):

```json
{
  "tipo_de_entrada": "arquitectura",
  "tecnologias": ["React", "CDN", "Django", "PostgreSQL", "Redis", "Celery", "RabbitMQ", "Kubernetes", "Argo CD"],
  "componentes_afectados": [],
  "nivel_de_criticidad": "baja",
  "resumen_tecnico": "Arquitectura de una SPA en React servida por CDN, comunicándose con API REST en Django, usando PostgreSQL, Redis, Celery con RabbitMQ, desplegada en Kubernetes gestionado por Argo CD."
}
```

**Caso ambiguo** ("A veces el sistema va lento por las tardes..."): el texto no nombra ninguna tecnología. El modelo recibe el error en cada reintento, pero como el prompt le prohíbe inventar datos (y el esquema rechaza rellenos como `N/A`), las tres respuestas fallan la validación y el resultado es un fallo controlado:

```
WARNING chain: intento descartado (InvalidOutputError): tecnologias: Value error, debe nombrar al menos una tecnología concreta
WARNING chain: intento descartado (InvalidOutputError): tecnologias: Value error, debe nombrar al menos una tecnología concreta
WARNING chain: intento descartado (InvalidOutputError): tecnologias: Value error, debe nombrar al menos una tecnología concreta

== Caso ambiguo ==
  extracción fallida [invalid_output]: tecnologias: Value error, debe nombrar al menos una tecnología concreta
```

**API key inválida**: falla al primer intento, sin reintentar:

```
  extracción fallida [provider]: HTTP 401: Invalid API Key
```

## Uso

```python
from chain import build_chain, create_llm, extract

chain = build_chain(create_llm())
outcome = await extract(chain, "ERROR [api] redis.exceptions.ConnectionError ...")
if outcome.analysis:
    print(outcome.analysis.model_dump_json(indent=2))
else:
    print(outcome.failure.kind, outcome.failure.message)
```

## Tests

```bash
uv run pytest
uv run mypy .
```


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


# Pre-entrega 4 - RAG híbrido en la nube con Pinecone

Evolución de la Pre-entrega 3 hacia la nube. Los documentos de PayFlow se dividen en chunks (una sección por chunk, con un techo de 600 tokens) y se suben a un índice **Pinecone Serverless**, con un namespace por categoría. Las preguntas se responden con **búsqueda híbrida**: vectorial en Pinecone más léxica con BM25, fusionadas con el `EnsembleRetriever` de LangChain. La calidad de la recuperación se mide con **Precision@k** y **Recall@k** sobre un Golden Dataset.

Los embeddings siguen siendo de **Gemini** (`gemini-embedding-001`), ahora a 1536 dimensiones. Por qué no OpenAI: ver [la ADR](docs/adr/0001-gemini-embeddings-at-1536.md).

## Estructura

```
schemas.py           # Category, EmbeddingSpace, ChunkingSettings, Chunk, ErrorCode, HybridWeights, GoldenCase
embedder.py          # GeminiEmbeddings (interfaz Embeddings de LangChain): asimétrico, normalizado, con reintentos
config.py            # variables de entorno, creación y validación del índice, manejo de errores de los scripts
ingest.py            # chunking por tokens, catálogo JSON por código, upsert por namespace
hybrid_retriever.py  # vectorial en todos los namespaces + BM25 + EnsembleRetriever
evaluate.py          # Precision@k y Recall@k: vectorial, BM25, tres pesos del híbrido y dos estrategias de chunking
golden_dataset.json  # 17 preguntas con sus documentos esperados
main.py              # demo: ingesta, búsquedas y evaluación
data/                # corpus de PayFlow: arquitectura, runbooks, logs y catálogo de errores
tests/               # pytest con embeddings falsos e InMemoryVectorStore, sin red
```

## Instalación

Con [uv](https://docs.astral.sh/uv/), desde `preentrega-4/`:

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

Hacen falta dos claves, las dos con plan gratuito:

- `GEMINI_API_KEY`: https://aistudio.google.com/apikey
- `PINECONE_API_KEY`: https://app.pinecone.io, en *API Keys*

El resto es opcional:

| Variable | Por defecto | Qué controla |
|---|---|---|
| `PINECONE_INDEX_NAME` | `payflow` | nombre del índice |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | modelo de embeddings |
| `CHUNK_TOKENS` | `600` | tamaño máximo de cada chunk, en tokens |
| `CHUNK_OVERLAP_TOKENS` | `80` | tokens compartidos entre chunks consecutivos |
| `HYBRID_VECTOR_WEIGHT` | `0.5` | peso de la búsqueda vectorial; BM25 recibe el resto |

## El índice en Pinecone

`ingest.py` crea el índice la primera vez, así que no hay que tocar la consola. Para crearlo a mano o replicarlo, los parámetros son:

| Parámetro | Valor |
|---|---|
| Tipo | Serverless |
| Nube y región | `aws` / `us-east-1` (la única del plan gratuito) |
| Dimensión | `1536` |
| Métrica | `cosine` |
| Tag | `embedding_model=gemini-embedding-001` |

Si el índice ya existe con otra dimensión, otra métrica u otro modelo, los scripts se niegan a usarlo en vez de mezclar vectores que no se pueden comparar:

```
El índice 'payflow' se creó con gemini-embedding-001 (768 dims, cosine) y ahora se pide gemini-embedding-001 (1536 dims, cosine). Borralo desde la consola de Pinecone o usá otro PINECONE_INDEX_NAME.
```

## Ejecución

Cada paso se puede correr por separado:

```bash
uv run python ingest.py      # carga data/ en Pinecone
uv run python evaluate.py    # Precision@k y Recall@k sobre el índice + comparación de chunking
uv run python main.py        # ingesta + búsquedas de demo + evaluación
```

## Ingesta

- **Categoría y namespace**: la categoría sale de la carpeta del archivo (`arquitectura`, `runbook`, `log`, `errores`) y cada categoría tiene su propio namespace en Pinecone. Un archivo fuera de esas carpetas se rechaza.
- **Chunking**: los `.md` primero se cortan por encabezado con `MarkdownHeaderTextSplitter`, así cada chunk sabe en qué sección está. Después cada sección, igual que los `.txt` y `.log`, pasa por `RecursiveCharacterTextSplitter.from_tiktoken_encoder` (`cl100k_base`) con un máximo de 600 tokens y 80 de solapamiento. Una sección corta queda como un solo chunk: no se fusiona con la siguiente para no mezclar temas.
- **Tamaño real de los chunks**: la mediana es de 78 tokens y el máximo de 434, por debajo de los 500-800 que sugiere la consigna. La razón es el corpus: el documento mediano tiene 288 tokens, así que un chunk de 500 sería el archivo entero. El techo de 600 recién corta cuando una sección es larga. La decisión está medida, no supuesta: ver [Chunking por sección vs. fusionado](#chunking-por-sección-vs-fusionado).
- **Catálogo de errores (JSON)**: cada código (`PF-5021`) es un chunk propio, con código, título, causa y runbook. Cortar el JSON por caracteres partiría los registros a la mitad.
- **Metadatos en Pinecone**: cada vector guarda todo lo necesario para mostrarlo, sin consultar otra base.

  | Campo | Ejemplo |
  |---|---|
  | `text` | el texto original del chunk |
  | `source` | `runbook/base-de-datos-no-responde.md` |
  | `category` | `runbook` |
  | `section` | `Mitigación`, o `PF-5021` en el catálogo (vacío en `.txt` y `.log`) |
  | `chunk_index` | `2` |

- **Idempotencia**: los IDs son determinísticos (`runbook/rollback-de-deploy.md#2`). Volver a ingestar un archivo pisa sus vectores con el upsert y después borra, por prefijo de ID, los que sobran. Así, si el archivo quedó más corto, no sobreviven chunks huérfanos, y la Source nunca queda sin vectores en el medio. Los archivos que ya no están en `data/` también se borran del índice. Si Gemini falla al pedir los embeddings, no se toca nada y la versión anterior sigue en el índice.
- **Consistencia eventual**: en Pinecone Serverless el upsert vuelve antes de que los vectores se puedan leer. Contar vectores no alcanza, porque una re-ingesta deja la misma cantidad. La ingesta espera hasta que cada namespace liste exactamente los IDs subidos y `fetch` devuelva el texto nuevo de cada uno.

## Búsqueda híbrida

```python
retrieval = pinecone_retrieval(index, embeddings)

retrieval.vector().invoke(pregunta)                          # solo Pinecone, todos los namespaces
retrieval.lexical().invoke(pregunta)                         # solo BM25
retrieval.hybrid(HybridWeights(vector=0.5)).invoke(pregunta) # EnsembleRetriever
retrieval.hybrid(weights, Category.RUNBOOK).invoke(pregunta) # las dos búsquedas, solo en un namespace
```

- **Vectorial**: un `PineconeVectorStore` por namespace. Sin categoría se consultan los cuatro y se mezclan los resultados por score. Las cuatro consultas usan el mismo vector de pregunta, que se calcula una sola vez.
- **BM25**: `BM25Retriever` se arma con los chunks que se leen del propio índice (`list` + `fetch`). Pinecone es la única fuente de verdad y los dos buscadores ven exactamente los mismos chunks. El texto se pasa a minúsculas y sin tildes, y se parte en tokens que conservan los guiones: `¿Mitigación del PF-5030 en pagos-api?` queda `mitigacion del pf-5030 en pagos-api`.
- **Fusión**: `EnsembleRetriever` combina los dos rankings con Reciprocal Rank Fusion ponderado. Cada buscador trae 20 chunks y los pesos se configuran con `HYBRID_VECTOR_WEIGHT`.

## Evaluación

`golden_dataset.json` tiene 17 preguntas con el formato del enunciado:

```json
{"pregunta": "procesador_timeout_corto", "documentos_esperados": ["runbook/cola-de-pagos-caida.md", "log/pagos-worker-2026-09-27.log"]}
```

Las preguntas cubren cuatro casos: paráfrasis semánticas (5), términos exactos como códigos, flags y comandos (6), preguntas que necesitan dos documentos (4) y nombres propios poco frecuentes (2).

- **La relevancia se mide por documento (Source), no por chunk.** El ranking de chunks se colapsa a documentos únicos en el orden en que aparecen, y las métricas se calculan sobre los k primeros. Así el dataset no se rompe si cambia el chunking.
- **Precision@k**: de los k primeros documentos, qué parte está entre los esperados. Siempre se divide por k.
- **Recall@k**: de los documentos esperados, qué parte aparece entre los k primeros.
- Antes de evaluar se verifica que todos los documentos esperados estén indexados, para que un error de tipeo en el dataset no pase por una falla de recuperación.

### Resultados

Ejecución real contra el índice (`uv run python evaluate.py`):

```
configuración       P@3     R@3     P@5     R@5
-----------------------------------------------
solo vectorial    0.451   0.912   0.306   1.000
solo BM25         0.412   0.833   0.259   0.863
híbrido 0.3/0.7   0.471   0.941   0.294   0.971
híbrido 0.5/0.5   0.490   0.971   0.294   0.971
híbrido 0.7/0.3   0.490   0.971   0.294   0.971
```

- **Top-3: gana el híbrido.** Con 0.5/0.5 recupera el 97 % de los documentos esperados, contra el 91 % del vectorial y el 83 % de BM25. Como la mayoría de las preguntas tiene uno o dos documentos esperados, el máximo posible de P@3 con este dataset es 0,510, y el híbrido queda en 0,490.
- **Top-5: el vectorial ya encuentra todo.** Llega a R@5 = 1,000 y a P@5 = 0,306, que es el techo de este dataset. El híbrido pierde una sola pregunta: *"¿Qué pasó el 28 de septiembre con los 503 de pagos-api...?"*. BM25 no pone ese log ni entre sus 8 primeros: el tokenizador deja `2026-09-28` como un único token, así que "28" no coincide, y "septiembre" no aparece en el log. En la fusión, sus resultados empujan el log fuera del top-5.
- **Pesos**: 0.5/0.5 y 0.7/0.3 empatan, y 0.3/0.7 (más peso a BM25) es peor. Por eso el valor por defecto es 0.5.
- **Dónde aporta BM25**: en términos exactos. Con `PF-5021`, el vectorial devuelve códigos de error parecidos (`PF-5022`, `PF-4001`), mientras que BM25 encuentra el log del incidente donde aparece ese código. El híbrido trae los dos.

### Chunking por sección vs. fusionado

Para comprobar si convenía acercarse a los 500-800 tokens de la consigna, `evaluate.py` repite la evaluación con una segunda estrategia: secciones consecutivas del mismo archivo fusionadas hasta 600 tokens. El catálogo de errores queda igual, un chunk por código. Las dos variantes corren en memoria (`InMemoryVectorStore`) con los mismos embeddings de Gemini; la variante por sección da exactamente los mismos números que Pinecone.

```
  por sección: 51 chunks, mediana 78 tokens, máximo 434
  fusionado: 25 chunks, mediana 243 tokens, máximo 434

configuración                    P@3     R@3     P@5     R@5
------------------------------------------------------------
por sección: vectorial         0.451   0.912   0.306   1.000
por sección: BM25              0.412   0.833   0.259   0.863
por sección: híbrido 0.5/0.5   0.490   0.971   0.294   0.971
fusionado: vectorial           0.431   0.892   0.271   0.922
fusionado: BM25                0.333   0.706   0.259   0.833
fusionado: híbrido 0.5/0.5     0.451   0.922   0.271   0.922
```

- **Fusionar empeora todas las métricas**, en los tres buscadores. BM25 es el que más pierde: en un chunk largo, un término exacto pesa menos frente a todo el resto del texto.
- **Ni fusionando se llega a 500 tokens**: la mediana queda en 243, porque la mayoría de los documentos es más corta que eso.
- **Esta tabla no mide la ventaja principal de los chunks chicos**: si los chunks recuperados se le pasan a un LLM, el top-5 por sección son unos 400 tokens de contexto, y el fusionado más de 1.200. La relevancia se mide por documento, así que eso no aparece en las métricas.

## Salida de ejemplo

Parte de `uv run python main.py`:

```
Ingestados 51 chunks (arquitectura=17, runbook=21, log=4, errores=9) en Pinecone

== Solo vectorial: PF-5021 ==
  1. [errores] errores/catalogo.json › PF-5021
  2. [errores] errores/catalogo.json › PF-5022
  3. [errores] errores/catalogo.json › PF-4001

== Solo BM25: PF-5021 ==
  1. [errores] errores/catalogo.json › PF-5021
  2. [log] log/pagos-worker-2026-09-27.log
  3. [errores] errores/catalogo.json › PF-7002

== Híbrido 0.5/0.5: PF-5021 ==
  1. [errores] errores/catalogo.json › PF-5021
  2. [errores] errores/catalogo.json › PF-7002
  3. [log] log/pagos-worker-2026-09-27.log

== Híbrido 0.5/0.5: ¿Qué hacemos si el procesador de tarjetas está lento y se acumulan pagos sin autorizar? ==
  1. [runbook] runbook/cola-de-pagos-caida.md › Mitigación
     ## Mitigación - Workers caídos: `kubectl rollout restart deployment/pagos-worker`. - Workers bloqueados [...]
  2. [errores] errores/catalogo.json › PF-5021
     PF-5021: Timeout del procesador de tarjetas Causa: Cobralia no respondió en 8 segundos. La transacción [...]
  3. [arquitectura] arquitectura/flujo-de-un-pago.md › Autorización
     ## Autorización `pagos-worker` toma el mensaje y llama al procesador de tarjetas con un timeout de 8 [...]

== Híbrido 0.5/0.5 solo en el namespace runbook: ¿Qué hacemos si el procesador de tarjetas está lento y se acumulan pagos sin autorizar? ==
  1. [runbook] runbook/cola-de-pagos-caida.md › Mitigación
  2. [runbook] runbook/cola-de-pagos-caida.md › Síntomas
  3. [runbook] runbook/cola-de-pagos-caida.md › Diagnóstico
```

La pregunta semántica mezcla el runbook que hay que seguir, el código de error y la parte de la arquitectura que explica el timeout. Limitada al namespace `runbook`, deja solo los pasos a seguir.

## Tests

```bash
uv run pytest
uv run mypy .
```

Los tests usan `FakeEmbeddings` (una bolsa de palabras hasheada) e `InMemoryVectorStore` de LangChain en lugar de Pinecone, así que no tocan la red. Cubren el chunking, las métricas y el retriever híbrido. `tests/test_cloud.py` sí llama a Gemini y a Pinecone, y se saltea si falta alguna de las dos claves.

# Pre-entrega 5 - Agente ReAct con memoria persistente

Agente de guardia para PayFlow construido con **LangGraph**. Recibe una pregunta, decide solo qué herramientas consultar (el catálogo de errores, los runbooks y los logs de los servicios), itera sobre lo que le devuelven y responde. El estado de cada conversación se guarda en **SQLite** con `AsyncSqliteSaver`, así que una conversación se puede retomar en otra ejecución del programa con su `thread_id`.

El modelo es `openai/gpt-oss-120b` en **Groq**. Por qué no OpenAI: ver [la ADR](docs/adr/0001-groq-instead-of-openai.md).

## Estructura

```
tools.py         # las tres herramientas (@tool) con sus esquemas Pydantic
agent.py         # AgentState, StateGraph (llm_node + ToolNode), checkpointer SQLite y run_turn
schemas.py       # modelos de la traza: TurnTrace, Step, ToolCallRecord, ToolResult
main.py          # demo de tres turnos y modo para retomar una conversación
trace.json       # traza de una ejecución real de la demo
data/            # catálogo de errores, runbooks y logs de PayFlow (copiados de la Pre-entrega 4)
tests/           # pytest con un modelo de chat guionado, sin red
```

## Instalación

Con [uv](https://docs.astral.sh/uv/), desde `preentrega-5/`:

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

Solo hace falta `GROQ_API_KEY` (gratuita en https://console.groq.com/keys). `GROQ_MODEL` es opcional y por defecto vale `openai/gpt-oss-120b`. Esa variable la comparten la P1, la P2 y la P5, así que si la cambiás tiene que ser un modelo con tool calling.

## Ejecución

```bash
uv run python main.py
```

Corre la demo: tres preguntas en una conversación nueva, imprime cada paso en consola y guarda la traza en `trace.json`. Al final muestra el comando para seguir esa misma conversación:

```bash
uv run python main.py --thread-id guardia-7e1f0e47 "Resumime en una línea qué códigos consultamos hasta ahora"
```

Es otro proceso, pero el agente recuerda lo anterior porque lo lee de `checkpoints.sqlite`:

```
Thread: guardia-7e1f0e47

== Resumime en una línea qué códigos consultamos hasta ahora ==
  [llm] razona: User asks: summarize in one line which codes we consulted so far. We consulted PF-5021 and [...]

Consultamos los códigos **PF‑5021** y **PF‑9999**.
```

Sin `--thread-id`, una pregunta suelta abre una conversación nueva: `uv run python main.py "¿Qué es PF-4290?"`.

- La demo escribe `trace.json`. Una pregunta suelta escribe `trace-<thread_id>.json`, así retomar una conversación no pisa la traza de la demo.
- `--thread-id` siempre va con una pregunta. Si el id no corresponde a ninguna conversación guardada (por ejemplo, por un error de tipeo), el programa lo avisa y termina en vez de empezar una conversación vacía.

## Herramientas

Las tres viven en `tools.py`. Cada una tiene un `args_schema` de Pydantic con la descripción de cada campo, que el modelo recibe junto con el docstring. Ninguna lanza excepciones: si algo falla devuelven un texto que empieza con `Error:` y explica qué valores son válidos, para que el modelo pueda corregirse en el paso siguiente.

| Herramienta | Argumentos | Devuelve | Si falla |
|---|---|---|---|
| `buscar_codigo_error` | `codigo` (`PF-` y cuatro dígitos; acepta minúsculas y espacios) | título, HTTP, causa y nombre del runbook | código inexistente: lista los válidos. Catálogo ilegible, que no es JSON o con entradas incompletas (se valida con Pydantic): lo informa |
| `leer_runbook` | `nombre` (`cola-de-pagos-caida`, también con carpeta o `.md`) | el runbook completo | nombre inexistente: lista los disponibles. No se puede salir de `data/runbook/` |
| `buscar_en_logs` | `texto`, `servicio` y `fecha` (`AAAA-MM-DD`), todos opcionales | las líneas que coinciden, ordenadas por hora, con un máximo de 20 | sin coincidencias: lista los servicios y fechas que tienen logs |

- `servicio` filtra por la etiqueta `[pagos-api]` de cada línea, no por el nombre del archivo: el log de `api-gateway` también tiene líneas de `pagos-api`.
- Si el modelo manda un argumento que no cumple el esquema (por ejemplo `codigo="5021"`), `ToolNode` le devuelve el error de validación como resultado de la herramienta y el modelo puede reintentar.

## El grafo

```
START → llm ──tools_condition──→ tools ─┐
         ↑   └──(sin tool calls)→ END   │
         └──────────────────────────────┘
```

- **Estado**: `AgentState` es un `TypedDict` con `messages: Annotated[list[AnyMessage], add_messages]`. El reducer agrega los mensajes nuevos de cada nodo a los anteriores en vez de reemplazarlos.
- **`llm`**: el modelo con `bind_tools` sobre las tres herramientas. El system prompt se antepone en cada llamada y no se guarda en el estado, así que cambiarlo no afecta conversaciones viejas.
- **`tools`**: `ToolNode` ejecuta las tool calls del último mensaje y agrega los `ToolMessage`.
- **Ciclo**: `tools_condition` va a `tools` si el modelo pidió herramientas y a `END` si respondió. Ningún `if` del código elige la herramienta.
- **Persistencia**: el grafo se compila con `AsyncSqliteSaver`. Cada invocación pasa `{"configurable": {"thread_id": ...}}` y el checkpointer guarda el estado después de cada paso.
- **Límite**: cada invocación lleva `recursion_limit=10`. Si el modelo no llega a una respuesta en 10 pasos, `run_turn` captura el `GraphRecursionError` y lo registra en la traza en vez de cortar el programa.
- **Fallas de Groq**: si Groq falla en medio de un turno, `run_turn` registra el error en la traza junto con los pasos que ya se habían hecho, y la demo sigue con la próxima pregunta. La traza se escribe aunque el programa se corte.
- **Asíncrono**: `llm_node` usa `ainvoke` y `run_turn` recorre el grafo con `astream(stream_mode="updates")`, que entrega un paso por vez para armar la traza.

## Traza

`trace.json` es la salida de una ejecución real de la demo. Tiene un objeto por turno con la pregunta, los pasos y la respuesta. Cada paso indica el nodo y, según corresponda, el razonamiento del modelo (`gpt-oss` lo devuelve aparte de la respuesta), las tool calls con sus argumentos, los resultados de las herramientas o el texto final.

El segundo turno muestra la memoria: la pregunta no nombra ningún código, pero el modelo busca `PF-5021` porque lo lee del turno anterior.

```json
{
  "thread_id": "guardia-7e1f0e47",
  "question": "¿Aparece en los logs? ¿A qué hora fue la primera vez?",
  "steps": [
    {
      "node": "llm",
      "reasoning": "We need to search logs for PF-5021. Use buscar_en_logs with texto \"PF-5021\". No date given.",
      "tool_calls": [{"name": "buscar_en_logs", "args": {"texto": "PF-5021"}}]
    },
    {
      "node": "tools",
      "tool_results": [{
        "name": "buscar_en_logs",
        "content": "2026-09-27 19:41:31 ERROR [pagos-worker] PF-5021 timeout de 8s esperando a Cobralia, transacción 87151 vuelve a la cola\n2026-09-27 19:41:33 ERROR [pagos-worker] PF-5021 timeout ..."
      }]
    },
    {
      "node": "llm",
      "content": "En los logs de **pagos‑worker** aparece el error **PF‑5021**. La primera aparición registrada es: **2026‑09‑27 19:41:31** ..."
    }
  ],
  "answer": "En los logs de **pagos‑worker** aparece el error **PF‑5021**. ..."
}
```

Los tres turnos de la demo:

1. **"¿Qué significa el error PF-5021 y qué tengo que hacer?"**: el modelo llama a `buscar_codigo_error`, lee en el resultado que el runbook es `cola-de-pagos-caida`, llama a `leer_runbook` y responde con los pasos. Son dos vueltas del ciclo en un mismo turno.
2. **"¿Aparece en los logs? ¿A qué hora fue la primera vez?"**: la prueba de memoria del ejemplo de arriba.
3. **"¿Y el PF-9999?"**: `buscar_codigo_error` devuelve `Error: el código PF-9999 no está en el catálogo. Códigos válidos: ...` y el modelo responde que ese código no existe, sin que el flujo se rompa.

## Tests

```bash
uv run pytest
uv run mypy .
```

Los tests de `test_agent.py` usan `ScriptedChatModel`, un modelo de chat que responde con una lista de mensajes fija y anota lo que recibe. No tocan la red. Cubren:

- un turno completo del ciclo: tool call, resultado y respuesta;
- la memoria: un turno escrito en SQLite se lee después de cerrar y reabrir el checkpointer, y otro `thread_id` no lo ve;
- el corte por `recursion_limit` con un modelo que nunca deja de pedir herramientas;
- la recuperación ante argumentos inválidos;
- una falla de Groq en medio del turno, que conserva los pasos ya hechos;
- que solo existen los threads con algún turno guardado.

`test_tools.py` prueba cada herramienta contra los datos reales y sus casos de error con datos rotos en una carpeta temporal.
