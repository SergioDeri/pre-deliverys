import argparse
import asyncio
import os
import sys
import textwrap
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from pydantic import TypeAdapter

from agent import compile_agent, create_llm, open_checkpointer, run_turn, thread_exists
from schemas import TurnTrace
from tools import payflow_tools

BASE_DIR = Path(__file__).parent
DEMO_QUESTIONS = (
    "¿Qué significa el error PF-5021 y qué tengo que hacer?",
    "¿Aparece en los logs? ¿A qué hora fue la primera vez?",
    "¿Y el PF-9999?",
)


def show(turn: TurnTrace) -> None:
    print(f"\n== {turn.question} ==")
    for step in turn.steps:
        if step.reasoning:
            print(f"  [{step.node}] razona: {textwrap.shorten(step.reasoning, width=100)}")
        for call in step.tool_calls:
            print(f"  [{step.node}] llama a {call.name}({call.args})")
        for result in step.tool_results:
            print(f"  [{step.node}] {result.name} -> {textwrap.shorten(result.content, width=100)}")
    if turn.error:
        print(f"  corte: {turn.error}")
    if turn.answer:
        print(f"\n{turn.answer}")


async def main(thread_id: str, questions: list[str], trace_file: Path, resume: bool) -> None:
    turns: list[TurnTrace] = []
    async with open_checkpointer() as saver:
        if resume and not await thread_exists(saver, thread_id):
            sys.exit(f"No hay ninguna conversación guardada con el thread id '{thread_id}'.")
        print(f"Thread: {thread_id}")
        agent = compile_agent(create_llm(), payflow_tools(), saver)
        try:
            for question in questions:
                turn = await run_turn(agent, thread_id, question)
                show(turn)
                turns.append(turn)
        finally:
            trace_file.write_bytes(
                TypeAdapter(list[TurnTrace]).dump_json(turns, indent=2, exclude_defaults=True)
            )
    print(f"\nTraza guardada en {trace_file.name}. Para seguir esta conversación:")
    print(f'  uv run python main.py --thread-id {thread_id} "tu pregunta"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agente de guardia de PayFlow")
    parser.add_argument("pregunta", nargs="?", help="pregunta para un solo turno; sin ella corre la demo")
    parser.add_argument("--thread-id", help="retoma una conversación guardada en checkpoints.sqlite")
    args = parser.parse_args()
    if args.thread_id and not args.pregunta:
        parser.error("--thread-id necesita una pregunta")

    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("Falta GROQ_API_KEY: copiá .env.example a .env en la raíz y completalo.")
    thread_id = args.thread_id or f"guardia-{uuid4().hex[:8]}"
    if args.pregunta:
        asyncio.run(main(thread_id, [args.pregunta], BASE_DIR / f"trace-{thread_id}.json", bool(args.thread_id)))
    else:
        asyncio.run(main(thread_id, list(DEMO_QUESTIONS), BASE_DIR / "trace.json", resume=False))
