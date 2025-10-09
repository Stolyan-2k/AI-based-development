import pandas as pd
import matplotlib.pyplot as plt

def analyze_growth(df: pd.DataFrame, date_col: str, value_col: str):
    # Копия данных, сортировка по дате
    data = df.copy()
    data = data.sort_values(by=date_col)

    # Расчёт показателей
    data["Абс_прирост"] = data[value_col].diff()
    data["Темп_роста_%"] = (data[value_col] / data[value_col].shift(1)) * 100
    data["Темп_прироста_%"] = data["Темп_роста_%"] - 100

    # Средний темп роста
    avg_growth_rate = data["Темп_роста_%"].mean(skipna=True)

    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(data[date_col], data[value_col], marker='o', label=value_col)
    plt.plot(data[date_col], data["Абс_прирост"], marker='s', label="Абс_прирост")
    plt.plot(data[date_col], data["Темп_роста_%"], marker='^', label="Темп_роста, %")
    plt.plot(data[date_col], data["Темп_прироста_%"], marker='d', label="Темп_прироста, %")

    plt.title("Динамика показателя, прироста и темпов роста")
    plt.xlabel(date_col)
    plt.ylabel("Значение / %")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Проверка базового условия
    рост_положительный = (data["Абс_прирост"] > 0).all()

    print(f"Средний темп роста: {avg_growth_rate:.2f}%")
    print(f"Рост положительный: {рост_положительный}")

    return data, avg_growth_rate


# Пример использования
# import numpy as np

# np.random.seed(42)
# months = pd.date_range("2024-01-01", periods=12, freq="M")
# values = np.cumsum(np.random.randint(80, 120, size=12))
# df = pd.DataFrame({"Месяц": months, "Продажи": values})

# result_df, avg_rate = analyze_growth(df, "Месяц", "Продажи")