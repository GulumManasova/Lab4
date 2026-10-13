import json
import sys

from agent import run_agent, LOG_FILE, make_mock_llm


# Пять заранее подготовленных ответов модели.
MOCK_SCRIPT = [
    "Thought: Найду информацию о суточных.\n"
    "Action: text_search\n"
    "Action Input: суточные",

    "Thought: Проверю неизвестный инструмент.\n"
    "Action: weather\n"
    "Action Input: Бишкек",

    "Thought: Выполню вычисление.\n"
    "Action: calculator\n"
    "Action Input: 1500 * 4",

    # Специально неправильный формат: нет Action и Final Answer.
    "Я сейчас не соблюдаю требуемый формат ответа.",

    "Thought: Все проверки выполнены.\n"
    "Final Answer: 6000 сом",
]


def read_steps(run_id):
    records = []
    with open(LOG_FILE, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("run_id") == run_id and record.get("event") == "step":
                records.append(record)
    return records


def test_offline():
    run_id = "offline_react_test"

    mock = make_mock_llm(MOCK_SCRIPT)

    answer = run_agent(
        "Сколько суточных за 4 дня?",
        llm=mock,
        run_id=run_id,
    )

    steps = read_steps(run_id)

    print("\n--- ОФЛАЙН-ТЕСТ ReAct ---")
    print("Итоговый ответ:", answer)
    print("Количество шагов:", len(steps))

    assert answer == "6000 сом", "Неверный итоговый ответ"
    assert len(steps) == 5, "Ожидалось 5 шагов"

    # Проверяем неизвестный инструмент.
    assert "неизвестный инструмент" in steps[1]["observation"].lower()

    # Проверяем вычисление.
    assert steps[2]["observation"] == "6000"

    # Проверяем реакцию на неправильный формат.
    assert "Нет 'Action:'" in steps[3]["observation"]
    assert steps[3]["tool_ok"] is False

    print("PASS: поиск и вызов инструмента")
    print("PASS: неизвестный инструмент")
    print("PASS: калькулятор")
    print("PASS: неправильный формат ответа")
    print("PASS: получение итогового ответа")
    print("\nВсе офлайн-проверки пройдены!")


if __name__ == "__main__":
    if "--real" in sys.argv:
        QUESTIONS = [
            ("Поиск", "Сколько календарных дней ежегодного отпуска у сотрудника?", "28"),
            ("Поиск", "Сколько дней в неделю можно работать удалённо?", "2"),
            ("Вычисление", "Сколько будет 2500 * 12 + 1800?", "31800"),
            ("Поиск + вычисление", "Сколько суточных получит сотрудник за командировку по стране на 6 дней?", "9000"),
            ("Поиск + вычисление", "Какая компенсация положена за служебную поездку на личном авто на 250 км?", "3000"),
        ]

        correct = 0
        for n, (kind, q, expected) in enumerate(QUESTIONS, 1):
            run_id = f"real{n}"
            answer = run_agent(q, run_id=run_id)
            ok = expected in answer.replace(" ", "")
            correct += ok
            print(f"\n#{n} [{kind}] {q}\nОтвет: {answer}\nВерно: {ok}")

        print(f"\nДоля верных: {correct}/{len(QUESTIONS)}")
    else:
        test_offline()
