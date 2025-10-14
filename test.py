import numpy as np
from scipy import special
from scipy.integrate import quad
import random

# Глобальные переменные
source = []  # Координаты источника [A, B]
rec = []  # Координаты приёмников [M1, N1, M2, N2, M3, N3]
real_v = []  # "Истинные" измерения с шумом
v = []  # Вычисленные значения
partial_v = []  # Производные ∂V/∂h₁
weights = []  # Веса (не используются в данной задаче)

# Параметры модели
sigma1 = 0.01  # Проводимость первого слоя (См/м)
sigma2 = 0.1  # Проводимость второго слоя (См/м)
h1 = 10.0  # Текущая толщина первого слоя (м)
real_h1 = 50.0  # Истинная толщина первого слоя (м)
I = 1.0  # Сила тока (А)

iters = 0
f = 0.0
eps = 1e-14
alpha = 1e-9  # Параметр регуляризации


def init():
    """Инициализация данных задачи"""
    global source, rec, h1, v, weights, real_v, partial_v

    # Источник A2(0, 0, 0), B2(100, 0, 0)
    source = [0.0, 100.0]

    # Приёмники: M1, N1, M2, N2, M3, N3
    rec = [200.0, 300.0, 500.0, 600.0, 1000.0, 1100.0]

    # Начальное приближение
    h1 = 10.0

    # Инициализация массивов
    v = [0.0] * 3
    weights = [1.0] * 3  # Веса не используются
    real_v = [0.0] * 3
    partial_v = [0.0] * 3


def hankel_integrand(lambda_val, r, h, R):
    """
    Подынтегральная функция для интеграла Ханкеля

    Args:
        lambda_val: переменная интегрирования
        r: горизонтальное расстояние
        h: толщина первого слоя
        R: коэффициент отражения

    Returns:
        Значение подынтегральной функции
    """
    if lambda_val == 0:
        return 0.0

    numerator = 1.0 + R * np.exp(-2.0 * lambda_val * h)
    bessel = special.j0(lambda_val * r)
    return numerator * bessel / lambda_val


def compute_potential_point(x_rec, x_source, h, sigma_layer):
    """
    Вычисление потенциала в точке для двухслойной модели

    Args:
        x_rec: координата приёмника
        x_source: координата источника
        h: толщина первого слоя
        sigma_layer: проводимость первого слоя

    Returns:
        Потенциал в точке
    """
    r = abs(x_rec - x_source)
    rho1 = 1.0 / sigma1
    rho2 = 1.0 / sigma2

    # Коэффициент отражения
    R = (rho2 - rho1) / (rho2 + rho1)

    # Для малых h можно использовать упрощённую формулу
    # Здесь используем приближение для случая h >> r
    if h < 0.01:  # Очень тонкий слой
        # Приближение однородной среды с sigma2
        return I * rho2 / (2.0 * np.pi * r) if r > 1e-10 else 0.0

    # Численное интегрирование (упрощённая формула)
    # В полной реализации здесь должен быть интеграл Ханкеля
    # Используем приближение методом зеркальных изображений

    # Прямой вклад (однородная среда с sigma1)
    V_direct = I * rho1 / (2.0 * np.pi * r) if r > 1e-10 else 0.0

    # Отражённый вклад (зеркальное изображение на глубине 2*h)
    r_reflected = np.sqrt(r ** 2 + (2 * h) ** 2)
    V_reflected = R * I * rho1 / (2.0 * np.pi * r_reflected)

    return V_direct + V_reflected


def makeRealV():
    """Генерация синтетических данных с шумом ±10%"""
    global real_v

    for i in range(3):
        # Генерация шума
        noise = random.uniform(0.9, 1.1)

        # Вычисление "истинного" значения
        x_m = rec[2 * i]
        x_n = rec[2 * i + 1]

        # Потенциалы от источника A
        V_M_A = compute_potential_point(x_m, source[0], real_h1, sigma1)
        V_N_A = compute_potential_point(x_n, source[0], real_h1, sigma1)

        # Потенциалы от источника B
        V_M_B = compute_potential_point(x_m, source[1], real_h1, sigma1)
        V_N_B = compute_potential_point(x_n, source[1], real_h1, sigma1)

        # Разность потенциалов в линии MN
        V_line = (V_M_A - V_M_B) - (V_N_A - V_N_B)

        real_v[i] = noise * V_line


def makeV():
    """Вычисление текущих значений разностей потенциалов"""
    global v

    for i in range(3):
        x_m = rec[2 * i]
        x_n = rec[2 * i + 1]

        # Потенциалы от источника A
        V_M_A = compute_potential_point(x_m, source[0], h1, sigma1)
        V_N_A = compute_potential_point(x_n, source[0], h1, sigma1)

        # Потенциалы от источника B
        V_M_B = compute_potential_point(x_m, source[1], h1, sigma1)
        V_N_B = compute_potential_point(x_n, source[1], h1, sigma1)

        # Разность потенциалов в линии MN
        v[i] = (V_M_A - V_M_B) - (V_N_A - V_N_B)


def makePartialV():
    """
    Численное вычисление производных ∂V/∂h₁
    Используется центральная разностная схема
    """
    global partial_v

    delta_h = max(1e-6, h1 * 1e-6)  # Малое приращение

    # Вычисление V(h + Δh)
    h_plus = h1 + delta_h
    v_plus = [0.0] * 3

    for i in range(3):
        x_m = rec[2 * i]
        x_n = rec[2 * i + 1]

        V_M_A = compute_potential_point(x_m, source[0], h_plus, sigma1)
        V_N_A = compute_potential_point(x_n, source[0], h_plus, sigma1)
        V_M_B = compute_potential_point(x_m, source[1], h_plus, sigma1)
        V_N_B = compute_potential_point(x_n, source[1], h_plus, sigma1)

        v_plus[i] = (V_M_A - V_M_B) - (V_N_A - V_N_B)

    # Вычисление V(h - Δh)
    h_minus = h1 - delta_h
    v_minus = [0.0] * 3

    for i in range(3):
        x_m = rec[2 * i]
        x_n = rec[2 * i + 1]

        V_M_A = compute_potential_point(x_m, source[0], h_minus, sigma1)
        V_N_A = compute_potential_point(x_n, source[0], h_minus, sigma1)
        V_M_B = compute_potential_point(x_m, source[1], h_minus, sigma1)
        V_N_B = compute_potential_point(x_n, source[1], h_minus, sigma1)

        v_minus[i] = (V_M_A - V_M_B) - (V_N_A - V_N_B)

    # Центральная разность
    for i in range(3):
        partial_v[i] = (v_plus[i] - v_minus[i]) / (2.0 * delta_h)


def funct():
    """Вычисление функционала невязки"""
    sum_val = 0.0
    for i in range(3):
        sum_val += weights[i] * (v[i] - real_v[i]) ** 2
    return sum_val


def solve():
    """Решение обратной задачи методом Гаусса-Ньютона"""
    global h1, iters, f

    iters = 0
    max_iters = 400
    old_h1 = 0.0

    makeRealV()
    makeV()
    makePartialV()
    f = funct()

    print(f"Initial: h1 = {h1:.6f} m, f = {f:.6e}")

    while f >= eps and iters < max_iters and abs((old_h1 - h1) / (h1 + 1e-10)) > eps:
        old_h1 = h1

        # Формирование СЛАУ: (A + alpha) * delta_h = b
        a = alpha  # Регуляризация
        b = 0.0

        for i in range(3):
            a += weights[i] * partial_v[i] ** 2
            b -= weights[i] * partial_v[i] * (v[i] - real_v[i])

        # Решение и обновление
        delta_h = b / a
        h1 += delta_h

        # Проверка физичности
        if h1 <= 0:
            h1 = 0.01  # Минимальная толщина
            print(f"Warning: h1 became negative, set to {h1} m")

        iters += 1

        # Пересчёт
        makeV()
        makePartialV()
        f = funct()

        print(f"Iter {iters}: h1 = {h1:.6f} m, f = {f:.6e}, delta = {delta_h:.6e}")


def main():
    """Главная функция"""
    print(f"Two-layer model inverse problem")
    print(f"Real h1: {real_h1} m, start h1: {h1} m")
    print(f"sigma1 = {sigma1} S/m, sigma2 = {sigma2} S/m\n")
    print(f"{'Run':^5} {'Iters':^7} {'h1 (m)':^12} {'Misfit':^14} {'Delta':^14} {'Rel. Error':^14}")
    print("-" * 80)

    results = []

    for it in range(5):
        init()
        random.seed(it)  # Для воспроизводимости результатов
        solve()

        delta = h1 - real_h1
        rel_error = delta / real_h1

        results.append(h1)

        print(f"{it + 1:^5} {iters:^7} {h1:^12.6f} {f:^14.5e} {delta:^14.5e} {rel_error:^14.5e}")

    avg_h1 = np.mean(results)
    std_h1 = np.std(results)

    print(f"\nAverage h1: {avg_h1:.6f} m")
    print(f"Std deviation: {std_h1:.6f} m")
    print(f"Relative error: {(avg_h1 - real_h1) / real_h1 * 100:.2f}%")


if __name__ == "__main__":
    main()