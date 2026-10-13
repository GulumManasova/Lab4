import json
import re
import time
from datetime import datetime, timezone

from openai import OpenAI
from tools import calculator, text_search

# ---------- Настройки: локальная Ollama ----------
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)
MODEL = "qwen3.5:4b"
LOG_FILE = "agent_log.jsonl"

TOOLS = {"calculator": calculator, "text_search": text_search}

SYSTEM_PROMPT = """Ты — ассистент, который решает задачи пошагово в формате ReAct.
Доступные инструменты:
{tools}

Строго соблюдай формат. На каждом шаге пиши:
Thought: рассуждение о том, что делать дальше
Action: имя_инструмента
Action Input: входные данные инструмента одной строкой
После этого ОСТАНОВИСЬ и жди строку Observation с результатом.

Когда информации достаточно, напиши:
Thought: итоговое рассуждение
Final Answer: ответ пользователю

Правила: не придумывай Observation сам; используй только перечисленные инструменты;
все вычисления делай через calculator; факты о компании бери только из text_search.
Если в документах нет ответа, так и скажи в Final Answer.

Пример:
Question: Сколько стоит 3 билета по 200 сом?
Thought: Нужно умножить.
Action: calculator
Action Input: 3 * 200
Observation: 600
Thought: Готово.
Final Answer: 600 сом"""

# ---------- Парсер ----------
ACTION_RE = re.compile(r"Action:\s*([\w\-]+)\s*\n\s*Action Input:\s*(.+)")
FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.S)


def parse(text: str):
    """('final', answer) | ('action', name, arg) | ('error', message)"""
    text = text.split("Observation:")[0]
    m_act, m_fin = ACTION_RE.search(text), FINAL_RE.search(text)
    if m_fin and (not m_act or m_fin.start() < m_act.start()):
        return ("final", m_fin.group(1).strip())
    if m_act:
        return ("action", m_act.group(1).strip(), m_act.group(2).strip())
    return ("error", "Нет 'Action:'+'Action Input:' и нет 'Final Answer:'. Соблюдай формат.")


# ---------- Вызов модели и журнал ----------
def call_llm(messages):
    """Возвращает (текст, usage-словарь)."""
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0, stop=["Observation:"])
    text = (resp.choices[0].message.content or "").strip()
    usage = {}
    if resp.usage:
        usage = {"prompt_tokens": resp.usage.prompt_tokens,
                 "completion_tokens": resp.usage.completion_tokens,
                 "total_tokens": resp.usage.total_tokens}
    return text, usage


def make_mock_llm(script):
    """Подставная модель: отладка без Ollama."""
    replies = iter(script)
    return lambda messages: (next(replies), {"prompt_tokens": 0, "completion_tokens": 0,
                                             "total_tokens": 0})


def log(record: dict):
    record = {"ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **record}
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------- Цикл ReAct ----------
def run_agent(question: str, max_steps: int = 8, llm=call_llm, run_id: str = None) -> str:
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    tools_desc = ("- calculator: вычисляет арифметическое выражение (например, 1500 * 4)\n"
                  "- text_search: ищет фрагменты во внутренних документах компании")
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(tools=tools_desc)},
                {"role": "user", "content": f"Question: {question}"}]
    log({"run_id": run_id, "event": "start", "question": question, "model": MODEL})

    for step in range(1, max_steps + 1):
        t0 = time.perf_counter()
        text, usage = llm(messages)
        latency = round(time.perf_counter() - t0, 3)
        parsed = parse(text)

        record = {"run_id": run_id, "step": step, "event": "step", "llm_output": text,
                  "action": None, "action_input": None, "observation": None,
                  "tool_ok": None, "latency_s": latency, "usage": usage}

        if parsed[0] == "final":
            log(record)
            log({"run_id": run_id, "event": "final", "step": step, "answer": parsed[1]})
            return parsed[1]

        if parsed[0] == "action":
            _, name, arg = parsed
            record["action"], record["action_input"] = name, arg
            if name in TOOLS:
                observation = TOOLS[name](arg)
                record["tool_ok"] = not observation.startswith(("Ошибка", "Ничего не найдено"))
            else:
                observation = (f"Ошибка: неизвестный инструмент '{name}'. "
                               f"Доступные: {', '.join(TOOLS)}")
                record["tool_ok"] = False
        else:  # error формата
            observation = parsed[1]
            record["tool_ok"] = False

        record["observation"] = observation
        log(record)
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    log({"run_id": run_id, "event": "max_steps_reached", "step": max_steps})
    return "Не удалось получить ответ за отведённое число шагов."


if __name__ == "__main__":
    print(run_agent(input("Вопрос: ")))