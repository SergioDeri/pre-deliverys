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
