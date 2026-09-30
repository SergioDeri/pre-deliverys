from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

CRITICALITY_CRITERIA = """\
- alta: hay caída del servicio, pérdida o corrupción de datos, errores en cascada o un fallo de seguridad.
- media: hay degradación visible (lentitud, errores intermitentes) o un riesgo claro que todavía no causa caída.
- baja: es una descripción sin fallos, o un problema menor sin impacto en usuarios."""

SYSTEM_TEMPLATE = """\
Eres un ingeniero de fiabilidad que analiza textos técnicos y extrae un análisis estructurado.

Reglas:
- Clasifica el texto como "arquitectura", "log_error" o "ambiguo". Usa "ambiguo" cuando el texto \
no permita saber con seguridad qué sistema describe ni qué falla.
- Enumera solo tecnologías nombradas de forma explícita en el texto. No las deduzcas ni las inventes.
- En un log de error indica siempre qué componentes están afectados.
- Un texto ambiguo nunca tiene criticidad alta.

Criterios de criticidad:
{criticality_criteria}

Escribe el resumen técnico en {output_language}."""

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEMPLATE),
        ("human", "Texto a analizar:\n\n{input_text}"),
        MessagesPlaceholder("correction", optional=True),
    ]
).partial(criticality_criteria=CRITICALITY_CRITERIA, output_language="español")

CORRECTION_TEMPLATE = """\
Tu respuesta anterior fue rechazada: {error}
Vuelve a llamar a la herramienta corrigiendo ese problema, sin inventar datos que el texto no contiene.
Si la respuesta se cortó, sé más breve."""


def correction_messages(error: Exception) -> list[BaseMessage]:
    return [HumanMessage(CORRECTION_TEMPLATE.format(error=error))]
