import re
from datetime import date, datetime, time
from functools import reduce
from pathlib import Path
from typing import Annotated, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    Field,
    TypeAdapter,
    ValidationError,
    WithJsonSchema,
    field_validator,
)

from app.agents.specialist import compile_specialist, delegate
from app.schemas import ErrorCodeEntry, Evidence, LogLine
from app.state import Node, OrchestratorState, merge_evidence

DATA_DIR = Path(__file__).parent.parent.parent / "data"
MAX_LOG_LINES = 20
LOG_LINE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(?P<level>\w+)\s+\[(?P<service>[^\]]+)\]\s+(?P<message>.*)$"
)

PROMPT = """\
Sos el investigador de un equipo que analiza incidentes de PayFlow, una plataforma de pagos. Tu \
trabajo es juntar evidencia con tus herramientas: logs de los servicios, catálogo de errores, \
runbooks y documentos de arquitectura. Empezá por los logs filtrando solo por fecha y franja \
horaria (desde/hasta), sin servicio ni texto, para traer todo lo que pasó alrededor del horario \
del incidente en todos los servicios: la causa puede estar en uno que la pregunta no nombra. El \
texto busca palabras exactas, no horarios. No repitas una búsqueda que ya hiciste. Si aparece un código PF-, \
buscalo en el catálogo. Si la pregunta pide qué hacer, leé el runbook que corresponda a los \
síntomas. No saques conclusiones ni hagas cuentas: de eso se encarga el analista. Al terminar, \
contá qué encontraste citando las líneas de log relevantes tal cual, el runbook que leíste con sus \
pasos de mitigación, y qué buscaste sin resultado. Respondé en español."""


def clock_part(value: object) -> object:
    return value.replace("T", " ").split(" ")[-1] if isinstance(value, str) else value


ClockTime = Annotated[
    time,
    BeforeValidator(clock_part),
    AfterValidator(lambda value: value.replace(tzinfo=None)),
    WithJsonSchema({"type": "string"}),
]

CATALOG = TypeAdapter(list[ErrorCodeEntry])


class ErrorCodeQuery(BaseModel):
    codigo: str = Field(
        pattern=r"^PF-\d{4}$", description="Código de error de PayFlow, por ejemplo PF-5021."
    )

    @field_validator("codigo", mode="before")
    @classmethod
    def normalize(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value


class RunbookQuery(BaseModel):
    nombre: str = Field(
        description="Nombre del runbook tal como lo devuelve buscar_codigo_error, por ejemplo cola-de-pagos-caida."
    )


class ArchitectureQuery(BaseModel):
    documento: str = Field(
        description="Documento de arquitectura: vision-general, flujo-de-un-pago, capacidad, webhooks o conciliacion."
    )


class LogQuery(BaseModel):
    texto: str = Field(
        default="", description="Texto a buscar dentro de la línea, sin distinguir mayúsculas."
    )
    servicio: str | None = Field(
        default=None, description="Servicio que escribió la línea, por ejemplo pagos-api o pgbouncer."
    )
    fecha: Annotated[date, WithJsonSchema({"type": "string"})] | None = Field(
        default=None, description="Día de la línea, en formato AAAA-MM-DD."
    )
    desde: ClockTime | None = Field(default=None, description="Hora mínima de la línea, en formato HH:MM o HH:MM:SS.")
    hasta: ClockTime | None = Field(default=None, description="Hora máxima de la línea, en formato HH:MM o HH:MM:SS.")


def researcher_tools(data_dir: Path = DATA_DIR) -> list[BaseTool]:
    catalog_path = data_dir / "errores" / "catalogo.json"
    runbook_dir = data_dir / "runbook"
    architecture_dir = data_dir / "arquitectura"
    log_dir = data_dir / "log"

    @tool(args_schema=ErrorCodeQuery, response_format="content_and_artifact")
    def buscar_codigo_error(codigo: str) -> tuple[str, Evidence]:
        """Busca un código de error de PayFlow (PF-XXXX) en el catálogo y devuelve su título,
        estado HTTP, causa y el runbook a seguir. Usala cuando un log o la pregunta mencionen un código PF-."""
        try:
            entries = CATALOG.validate_json(catalog_path.read_bytes())
        except (OSError, ValidationError) as exc:
            return f"Error: no se pudo leer el catálogo de errores ({exc}).", Evidence()
        entry = next((e for e in entries if e.codigo == codigo), None)
        if entry is None:
            valid = ", ".join(e.codigo for e in entries)
            return f"Error: el código {codigo} no está en el catálogo. Códigos válidos: {valid}.", Evidence()
        runbook = Path(entry.runbook).stem if entry.runbook else "ninguno"
        text = (
            f"{entry.codigo}: {entry.titulo}\n"
            f"HTTP: {entry.http or 'no aplica'}\n"
            f"Causa: {entry.causa}\n"
            f"Runbook: {runbook}"
        )
        return text, Evidence(error_codes=[entry])

    @tool(args_schema=RunbookQuery, response_format="content_and_artifact")
    def leer_runbook(nombre: str) -> tuple[str, Evidence]:
        """Devuelve el texto completo de un runbook de PayFlow: síntomas, diagnóstico y mitigación.
        Usala con el nombre que indica el catálogo de errores o cuando los síntomas coincidan con uno."""
        return read_document(runbook_dir, nombre, "runbook"), Evidence()

    runbooks = ", ".join(sorted(path.stem for path in runbook_dir.glob("*.md")))
    leer_runbook.description += f" Runbooks disponibles: {runbooks}."

    @tool(args_schema=ArchitectureQuery, response_format="content_and_artifact")
    def leer_arquitectura(documento: str) -> tuple[str, Evidence]:
        """Devuelve un documento de arquitectura de PayFlow: componentes, de qué depende cada servicio,
        límites de capacidad. Usala para saber cómo se propaga una falla de un servicio a otro."""
        return read_document(architecture_dir, documento, "documento de arquitectura"), Evidence()

    @tool(args_schema=LogQuery, response_format="content_and_artifact")
    def buscar_en_logs(
        texto: str = "",
        servicio: str | None = None,
        fecha: date | None = None,
        desde: time | None = None,
        hasta: time | None = None,
    ) -> tuple[str, Evidence]:
        """Busca líneas en los logs de los servicios de PayFlow. Usala para saber cuándo, dónde y
        con qué nivel apareció un error, un código PF- o un síntoma."""
        try:
            all_lines = sorted(
                (line for path in log_dir.glob("*.log") for line in read_log(path)),
                key=lambda line: line.timestamp,
            )
        except (OSError, UnicodeDecodeError) as exc:
            return f"Error: no se pudieron leer los logs ({exc}).", Evidence()
        lines = [
            line
            for line in all_lines
            if texto.lower() in str(line).lower()
            and (servicio is None or line.service == servicio.strip().lower())
            and (fecha is None or line.timestamp.date() == fecha)
            and (desde is None or line.timestamp.time() >= desde)
            and (hasta is None or line.timestamp.time() <= hasta)
        ]
        if not lines:
            services = ", ".join(sorted({line.service for line in all_lines}))
            dates = ", ".join(sorted({line.timestamp.date().isoformat() for line in all_lines}))
            return (
                "Sin resultados para esos filtros. "
                f"Servicios con logs: {services}. Fechas con logs: {dates}.",
                Evidence(),
            )
        shown = lines[:MAX_LOG_LINES]
        text = "\n".join(str(line) for line in shown)
        if len(lines) > MAX_LOG_LINES:
            text += (
                f"\n[{len(lines)} líneas coinciden; se muestran las primeras {MAX_LOG_LINES}, "
                "pero el analista recibe todas.]"
            )
        return text, Evidence(log_lines=lines)

    return [buscar_codigo_error, leer_runbook, leer_arquitectura, buscar_en_logs]


def researcher_node(llm: BaseChatModel, data_dir: Path = DATA_DIR) -> Node:
    specialist = compile_specialist(llm, researcher_tools(data_dir), PROMPT)

    async def researcher(state: OrchestratorState) -> dict[str, Any]:
        messages = await delegate(specialist, "researcher", state, Evidence())
        collected = [m.artifact for m in messages if isinstance(m, ToolMessage) and isinstance(m.artifact, Evidence)]
        return {
            "messages": [AIMessage(messages[-1].text, name="researcher")],
            "evidence": reduce(merge_evidence, collected, Evidence()),
            "sender": "researcher",
        }

    return researcher


def read_document(folder: Path, name: str, kind: str) -> str:
    documents = {path.stem: path for path in folder.glob("*.*")}
    stem = Path(name.strip()).stem
    if stem not in documents:
        return f"Error: no existe el {kind} '{name}'. Disponibles: {', '.join(sorted(documents))}."
    try:
        return documents[stem].read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return f"Error: no se pudo leer el {kind} '{stem}' ({exc})."


def read_log(path: Path) -> list[LogLine]:
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        match = LOG_LINE.match(raw)
        if match is None:
            continue
        try:
            lines.append(
                LogLine(
                    timestamp=datetime.fromisoformat(match["timestamp"]),
                    level=match["level"],
                    service=match["service"],
                    message=match["message"],
                )
            )
        except ValidationError:
            continue
    return lines
