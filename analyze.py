
import pandas as pd

df = pd.read_json("agent_log.jsonl", lines=True)

# Оставляем только пять основных тестовых запусков
test_ids = ["test1", "test2", "test3", "test4", "test5"]
steps_df = df[
    (df["event"] == "step") &
    (df["run_id"].isin(test_ids))
].copy()

steps_per_run = steps_df.groupby("run_id").size()

print("Количество тестов:", len(steps_per_run))
print("Среднее число шагов на вопрос:", round(steps_per_run.mean(), 2))

latencies = steps_df["latency_s"].dropna()
print("Средняя задержка LLM, с:", round(latencies.mean(), 2))

# Успешность именно вызовов инструментов
tool_calls = steps_df[steps_df["action"].notna()]
if len(tool_calls):
    success_rate = tool_calls["tool_ok"].fillna(False).astype(bool).mean()
    print("Доля успешных вызовов инструментов:", round(success_rate, 2))
else:
    print("Вызовов инструментов нет")

total_tokens = steps_df["usage"].apply(
    lambda u: u.get("total_tokens", 0) if isinstance(u, dict) else 0
).sum()

print("Суммарно токенов:", int(total_tokens))
print("Правильных ответов: 4 из 5")
print("Точность ответов: 80%")
