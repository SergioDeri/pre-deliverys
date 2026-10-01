import argparse
import asyncio
import os
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv

from agents.researcher import researcher_tools
from agents.specialist import compile_specialist
from graph import build_graph, create_llm, run_investigation
from schemas import Contribution, Delegation, DelegationTrace

BASE_DIR = Path(__file__).parent
DEMO_QUESTION = (
    "El 28/09/2026 cerca de las 21:15 fallaron pagos. ¿Qué pasó, qué servicio lo originó, "
    "cuántos errores hubo en cada servicio, cuánto duró y qué hay que hacer si se repite?"
)


def show_step(step: Delegation | Contribution) -> None:
    if isinstance(step, Contribution):
        for call in step.tool_calls:
            print(f"  [{step.agent}] llama a {call.name}({call.args})")
        print(f"  [{step.agent}] aporta: {textwrap.shorten(step.content, width=110)}\n")
    else:
        override = " (override: todavía no había investigación)" if step.override else ""
        print(f"[supervisor] delega en {step.agent}{override}: {step.instruction}")
        print(f"             porque: {step.reason}")


def show_ending(trace: DelegationTrace) -> None:
    if trace.finish_reason:
        print(f"[supervisor] termina: {trace.finish_reason}")
    print(f"\nRecorrido: {' -> '.join([*trace.path, 'END'])}")
    if trace.error:
        print(f"Corte: {trace.error}")
    if trace.final_answer:
        print(f"\n{trace.final_answer}")


def mermaid() -> None:
    llm = create_llm()
    print("Orquestador:")
    print(build_graph(llm).get_graph().draw_mermaid())
    print("Cada especialista por dentro:")
    print(compile_specialist(llm, researcher_tools(), "").get_graph().draw_mermaid())


async def main(question: str, trace_file: Path) -> None:
    print(f"== {question} ==\n")
    trace = await run_investigation(build_graph(create_llm()), question, on_step=show_step)
    show_ending(trace)
    trace_file.write_text(trace.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")
    print(f"\nTraza guardada en {trace_file.name}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orquestador multi-agente de incidentes de PayFlow")
    parser.add_argument("pregunta", nargs="?", help="pregunta a investigar; sin ella corre la demo")
    parser.add_argument("--mermaid", action="store_true", help="imprime el diagrama Mermaid del grafo")
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("Falta GROQ_API_KEY: copiá .env.example a .env en la raíz y completalo.")
    if args.mermaid:
        mermaid()
    elif args.pregunta:
        asyncio.run(main(args.pregunta, BASE_DIR / "trace-pregunta.json"))
    else:
        asyncio.run(main(DEMO_QUESTION, BASE_DIR / "trace.json"))
