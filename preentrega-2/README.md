# Pipeline de extracción validada — Pre-entrega 2

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
