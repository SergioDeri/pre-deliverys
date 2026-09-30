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
