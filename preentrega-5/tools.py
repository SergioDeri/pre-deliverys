import re
from datetime import date
from pathlib import Path

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field, TypeAdapter, ValidationError, field_validator

DATA_DIR = Path(__file__).parent / "data"
MAX_LOG_LINES = 20
LOG_LINE = re.compile(r"^(?P<fecha>\d{4}-\d{2}-\d{2}) \S+\s+\w+\s+\[(?P<servicio>[^\]]+)\]")


class ErrorCodeEntry(BaseModel):
    codigo: str
    titulo: str
    http: int | None
    causa: str
    runbook: str | None


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


class LogQuery(BaseModel):
    texto: str = Field(
        default="", description="Texto a buscar dentro de la línea, sin distinguir mayúsculas."
    )
    servicio: str | None = Field(
        default=None, description="Servicio que escribió la línea, por ejemplo pagos-api o pagos-worker."
    )
    fecha: date | None = Field(default=None, description="Día de la línea, en formato AAAA-MM-DD.")


def payflow_tools(data_dir: Path = DATA_DIR) -> list[BaseTool]:
    catalog_path = data_dir / "errores" / "catalogo.json"
    runbook_dir = data_dir / "runbook"
    log_dir = data_dir / "log"

    @tool(args_schema=ErrorCodeQuery)
    def buscar_codigo_error(codigo: str) -> str:
        """Busca un código de error de PayFlow (PF-XXXX) en el catálogo y devuelve su título,
        estado HTTP, causa y el runbook a seguir. Usala cuando la pregunta mencione un código PF-."""
        try:
            entries = CATALOG.validate_json(catalog_path.read_bytes())
        except (OSError, ValidationError) as exc:
            return f"Error: no se pudo leer el catálogo de errores ({exc})."
        entry = next((e for e in entries if e.codigo == codigo), None)
        if entry is None:
            valid = ", ".join(e.codigo for e in entries)
            return f"Error: el código {codigo} no está en el catálogo. Códigos válidos: {valid}."
        runbook = Path(entry.runbook).stem if entry.runbook else "ninguno"
        return (
            f"{entry.codigo}: {entry.titulo}\n"
            f"HTTP: {entry.http or 'no aplica'}\n"
            f"Causa: {entry.causa}\n"
            f"Runbook: {runbook}"
        )

    @tool(args_schema=RunbookQuery)
    def leer_runbook(nombre: str) -> str:
        """Devuelve el texto completo de un runbook de PayFlow: síntomas, diagnóstico y mitigación.
        Usala para saber qué hacer ante un incidente, con el nombre que indica el catálogo de errores."""
        available = sorted(path.stem for path in runbook_dir.glob("*.md"))
        name = Path(nombre.strip()).stem
        if name not in available:
            return f"Error: no existe el runbook '{nombre}'. Runbooks disponibles: {', '.join(available)}."
        try:
            return (runbook_dir / f"{name}.md").read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return f"Error: no se pudo leer el runbook '{name}' ({exc})."

    @tool(args_schema=LogQuery)
    def buscar_en_logs(texto: str = "", servicio: str | None = None, fecha: date | None = None) -> str:
        """Busca líneas en los logs de los servicios de PayFlow. Usala para saber cuándo y dónde
        apareció un error, un código PF- o un síntoma."""
        if not log_dir.is_dir():
            return f"Error: no existe la carpeta de logs {log_dir}."
        try:
            all_lines = sorted(
                line
                for path in log_dir.glob("*.log")
                for line in path.read_text(encoding="utf-8").splitlines()
            )
        except (OSError, UnicodeDecodeError) as exc:
            return f"Error: no se pudieron leer los logs ({exc})."
        lines = [
            line for line in all_lines if texto.lower() in line.lower() and _matches(line, servicio, fecha)
        ]
        if not lines:
            parsed = [match for line in all_lines if (match := LOG_LINE.match(line))]
            services = ", ".join(sorted({match["servicio"] for match in parsed}))
            dates = ", ".join(sorted({match["fecha"] for match in parsed}))
            return (
                "Sin resultados para esos filtros. "
                f"Servicios con logs: {services}. Fechas con logs: {dates}."
            )
        if len(lines) > MAX_LOG_LINES:
            notice = f"[{len(lines)} líneas coinciden; se muestran las primeras {MAX_LOG_LINES}. Afiná los filtros.]"
            return "\n".join([*lines[:MAX_LOG_LINES], notice])
        return "\n".join(lines)

    return [buscar_codigo_error, leer_runbook, buscar_en_logs]


def _matches(line: str, servicio: str | None, fecha: date | None) -> bool:
    if servicio is None and fecha is None:
        return True
    parsed = LOG_LINE.match(line)
    if parsed is None:
        return False
    if servicio is not None and parsed["servicio"] != servicio.strip().lower():
        return False
    return fecha is None or parsed["fecha"] == fecha.isoformat()
