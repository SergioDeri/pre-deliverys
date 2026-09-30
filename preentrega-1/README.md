# Cliente LLM asíncrono unificado — Pre-entrega 1

Capa de abstracción en Python 3.12 para hablar con modelos de lenguaje de forma no bloqueante, con respuesta completa o en streaming, validación con Pydantic y manejo de errores con reintentos.

> El enunciado propone OpenAI y Anthropic; este proyecto usa **Groq** como único proveedor. El motivo está en [docs/adr/0001-groq-only-provider.md](docs/adr/0001-groq-only-provider.md).

## Estructura

```
llm_client/
├── schemas.py       # ChatMessage, GenerationConfig, RetryPolicy, ModelResponse, LLMError
├── base.py          # BaseLLMClient: interfaz común, reintentos y streaming
├── groq_client.py   # GroqClient sobre AsyncGroq
├── errors.py        # LLMStreamError
└── factory.py       # create_client(): elige el proveedor según la configuración
main.py              # demo: respuesta completa, streaming y error controlado
tests/               # pytest, sin llamadas reales a la API
```

## Instalación

Con [uv](https://docs.astral.sh/uv/), desde `preentrega-1/`:

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

El `.env` se comparte con la Pre-entrega 2 y vive en la raíz del repositorio (ver el [README principal](../README.md)):

```bash
cp ../.env.example ../.env
```

Completa `GROQ_API_KEY` con una clave de https://console.groq.com/keys (el plan gratuito alcanza). `GROQ_MODEL` es opcional; por defecto se usa `openai/gpt-oss-20b`.

## Ejecución

```bash
uv run python main.py      # o: python main.py con el venv activado
```

La demo hace tres cosas:

1. Pide una respuesta completa y muestra tokens, latencia e intentos.
2. Repite la consulta en streaming, imprimiendo el texto a medida que llega.
3. Usa una API key inválida a propósito para mostrar que el error vuelve como un `LLMError` estructurado y el programa sigue.

## Uso

```python
from llm_client import ChatMessage, GenerationConfig, create_client

messages = [ChatMessage(role="user", content="Hola")]

async with create_client() as client:
    response = await client.generate(messages)
    if response.ok:
        print(response.content)
    else:
        print(response.error.kind, response.error.message)

    async for fragment in client.stream(messages, GenerationConfig(model="openai/gpt-oss-120b")):
        print(fragment, end="")
```

## Manejo de errores

- `generate()` nunca lanza por errores de la API: devuelve un `ModelResponse` con `error` cargado.
- Rate limit, timeouts, errores de red y 5xx se reintentan con backoff exponencial (1s, 2s, 4s + jitter), respetando `Retry-After` cuando el servidor lo envía. Los reintentos del SDK están desactivados para que no se sumen a los nuestros.
- Clave inválida o request mal formada se devuelven al instante, sin reintentar.
- En streaming solo se reintenta hasta recibir el primer fragmento; si la conexión se corta después, se lanza `LLMStreamError` con el `LLMError` dentro, para no repetir texto ya entregado.

## Agregar un proveedor

Crear una subclase de `BaseLLMClient` que implemente `_complete`, `_stream_fragments`, `_to_llm_error` y `aclose`, sumar el valor a `Provider` y registrarla en `factory.py`. El resto del código no cambia.

## Tests

```bash
uv run pytest
uv run mypy llm_client tests main.py
```
