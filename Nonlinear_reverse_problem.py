import numpy as np
from scipy.special import j0
from scipy.integrate import quad
from typing import List, Callable

# Глобальные переменные
source: List[float] = []  # Координаты источника [A, B]
rec: List[float] = []  # Координаты приёмников [M1, N1, M2, N2, M3, N3]
real_v: List[float] = []  # "Истинные" измерения
v: List[float] = []  # Вычисленные значения
partial_v: List[float] = []  # Производные ∂V/∂h₁
weights: List[float] = []  # Веса w_i = 1/|V_i|

# Параметры модели
sigma1: float = 0.01  # Проводимость первого слоя (См/м)
sigma2: float = 0.1  # Проводимость второго слоя (См/м)
h1: float = 10  # Текущая толщина первого слоя (м)
real_h1: float = 50.0  # Истинная толщина первого слоя (м)
I: float = 1.0  # Сила тока (А)

iters: int = 0
f: float = 0.0
eps: float = 1e-25
alpha: float = 1e-12 # Параметр регуляризации
receivers_count: int = 3
flag: str = 'H'
interg_flag: str = 'S'
noise_flag: bool = False  # Флаг для добавления шума
noise_level: float = 0.03  # Уровень шума (±3%)

def simpson_integration(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """
    Численное интегрирование методом Симпсона.
    Args:
        f: подынтегральная функция
        a: нижний предел
        b: верхний предел
        n: число интервалов (должно быть чётным)
    Returns:
        Приближение интеграла
    """
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        x = a + i * h
        if i % 2 == 0:
            s += 2 * f(x)
        else:
            s += 4 * f(x)
    return s * h / 3

def compute_potential_point_mirror(x_rec: float,
                                   x_source: float,
                                   h: float,
                                   sigma_layer: float) -> float:
    """
    Вычисление потенциала в точке для двухслойной модели
    Использует приближение методом зеркальных изображений

    Args:
        x_rec: координата приёмника
        x_source: координата источника
        h: толщина первого слоя
        sigma_layer: проводимость первого слоя

    Returns:
        Потенциал в точке
    """
    r: float = abs(x_rec - x_source)
    rho1: float = 1.0 / sigma1
    rho2: float = 1.0 / sigma2

    # Коэффициент отражения
    R: float = (rho2 - rho1) / (rho2 + rho1)

    if h < 1e-6:
        return I * rho2 / (2.0 * np.pi * r) if r > 1e-12 else 0.0

    # Приближение методом зеркальных изображений
    # Прямой вклад (однородная среда с sigma1)
    V_direct: float = I * rho1 / (2.0 * np.pi * r) if r > 1e-12 else 0.0

    # Отражённый вклад (зеркальное изображение на глубине 2*h)
    r_reflected: float = np.sqrt(r ** 2 + (2 * h) ** 2)
    V_reflected: float = R * I * rho1 / (2.0 * np.pi * r_reflected)

    return V_direct + V_reflected

def compute_potential_point_hankel(x_rec: float,
                                   x_source: float,
                                   h: float,
                                   sigma_layer: float,
                                   interg_flag: str = 'S') -> float:
    """
    Вычисление потенциала в точке для двухслойной модели с использованием интеграла Ханкеля.

    Args:
        x_rec: координата приёмника (x-координата)
        x_source: координата источника (x-координата)
        h: толщина первого слоя (h1)
        sigma_layer: проводимость первого слоя (sigma1)

    Returns:
        Потенциал в точке
    """
    global sigma1, sigma2, I, integral

    r: float = abs(x_rec - x_source)
    rho1: float = 1.0 / sigma1
    rho2: float = 1.0 / sigma2
    R: float = (sigma1 - sigma2) / (sigma1 + sigma2)  # Коэффициент отражения

    def integrand(lambda_: float) -> float:
        if lambda_ < 1e-12:
            return 0.0
        kernel: float = (1 + R * np.exp(-2 * lambda_ * h)) / (1 - R * np.exp(-2 * lambda_ * h)) / lambda_
        return j0(lambda_ * r) * kernel

    if r < 1e-12:
        return 0.0

    if h < 1e-6:
        return I * rho2 / (2.0 * np.pi * r) if r > 1e-12 else 0.0

    # Численное интегрирование
    if interg_flag == 'S':
        integral1, _ = quad(integrand, 0, 1, epsabs=1e-8, epsrel=1e-8, limit=50)
        integral2, _ = quad(integrand, 1, 10, epsabs=1e-8, epsrel=1e-8, limit=50)
        integral3, _ = quad(integrand, 10, 1000, epsabs=1e-8, epsrel=1e-8, limit=50)
        integral = integral1 + integral2 + integral3
    elif interg_flag == 'SIM':
        n = 100
        integral1 = simpson_integration(integrand, 0, 1, n)
        integral2 = simpson_integration(integrand, 1, 10, n)
        integral3 = simpson_integration(integrand, 10, 1000, n)
        integral = integral1 + integral2 + integral3

    return I * rho1 / (2.0 * np.pi) * integral

def init() -> None:
    """Инициализация данных задачи"""
    global source, rec, h1, v, weights, real_v, partial_v

    # Источник A2(0, 0, 0), B2(100, 0, 0)
    source = [0.0, 100.0]

    # Приёмники: M1(200,0,0), N1(300,0,0), M2(500,0,0), N2(600,0,0), M3(1000,0,0), N3(1100,0,0)
    rec = [200.0, 300.0, 500.0, 600.0, 1000.0, 1100.0]

    h1 = 10

    v = [0.0] * receivers_count
    weights = [0.0] * receivers_count
    real_v = [0.0] * receivers_count
    partial_v = [0.0] * receivers_count

def compute_line_voltage(x_m: float,
                         x_n: float,
                         x_a: float,
                         x_b: float,
                         h: float,
                         flag: str = 'H',
                         interg_flag: str = 'S') -> float:
    """
    Вычисление разности потенциалов в линии MN от источника AB

    Args:
        x_m, x_n: координаты приёмников M и N
        x_a, x_b: координаты источников A и B
        h: толщина первого слоя

    Returns:
        Разность потенциалов V_MN
    """
    global V_M_A, V_M_B, V_N_A, V_N_B

    if flag == 'H':
        # Потенциалы от источника A
        V_M_A = compute_potential_point_hankel(x_m, x_a, h, sigma1, interg_flag)
        V_N_A = compute_potential_point_hankel(x_n, x_a, h, sigma1, interg_flag)

        # Потенциалы от источника B
        V_M_B = compute_potential_point_hankel(x_m, x_b, h, sigma1, interg_flag)
        V_N_B = compute_potential_point_hankel(x_n, x_b, h, sigma1, interg_flag)
    elif flag == 'M':
        V_M_A = compute_potential_point_mirror(x_m, x_a, h, sigma1)
        V_N_A = compute_potential_point_mirror(x_n, x_a, h, sigma1)
        V_M_B = compute_potential_point_mirror(x_m, x_b, h, sigma1)
        V_N_B = compute_potential_point_mirror(x_n, x_b, h, sigma1)

    # Разность потенциалов в линии MN
    return (V_M_A - V_M_B) - (V_N_A - V_N_B)

def makeRealV(flag: str = 'H',
              interg_flag: str = 'S') -> None:
    """Генерация синтетических данных без шума"""
    global real_v, weights

    for i in range(receivers_count):
        x_m: float = rec[2 * i]
        x_n: float = rec[2 * i + 1]

        real_v[i] = compute_line_voltage(x_m, x_n, source[0], source[1], real_h1, flag, interg_flag)

        if noise_flag:
            noise = np.random.uniform(-noise_level, noise_level) * real_v[i]
            real_v[i] += noise

        # Веса w_i = 1/|V_i|
        weights[i] = 1.0 / abs(real_v[i]) if abs(real_v[i]) > 1e-10 else 1.0

    print(f"\n{'i':^5} {'V_i':^15} {'w_i':^15}")
    print("-" * 35)
    for i in range(receivers_count):
        print(f"{i + 1:^5} {real_v[i]:^15.6e} {weights[i]:^15.6e}")

def makeV(flag: str = 'H',
          interg_flag: str = 'S') -> None:
    """Вычисление текущих значений разностей потенциалов"""
    global v

    for i in range(receivers_count):
        x_m: float = rec[2 * i]
        x_n: float = rec[2 * i + 1]
        v[i] = compute_line_voltage(x_m, x_n, source[0], source[1], h1, flag, interg_flag)

def makePartialV(flag: str = 'H',
                 interg_flag: str = 'S') -> None:
    """
    Численное вычисление производных ∂V/∂h₁
    Используется центральная разностная схема
    """
    global partial_v

    delta_h: float = max(1e-6, h1 * 1e-6)

    # Вычисление V(h + Δh)
    v_plus: List[float] = [0.0] * receivers_count
    for i in range(receivers_count):
        x_m: float = rec[2 * i]
        x_n: float = rec[2 * i + 1]
        v_plus[i] = compute_line_voltage(x_m, x_n, source[0], source[1], h1 + delta_h, flag, interg_flag)

    # Вычисление V(h - Δh)
    v_minus: List[float] = [0.0] * receivers_count
    for i in range(receivers_count):
        x_m: float = rec[2 * i]
        x_n: float = rec[2 * i + 1]
        v_minus[i] = compute_line_voltage(x_m, x_n, source[0], source[1], h1 - delta_h, flag, interg_flag)

    # Центральная разность
    for i in range(receivers_count):
        partial_v[i] = (v_plus[i] - v_minus[i]) / (2.0 * delta_h)

def funct() -> float:
    """Вычисление взвешенного функционала невязки"""
    sum_val: float = 0.0
    for i in range(receivers_count):
        err: float = v[i] - real_v[i]
        sum_val += weights[i] * err * err
    return sum_val

def solve(flag: str = 'H',
          interg_flag: str = 'S') -> None:
    """Решение обратной задачи методом Гаусса-Ньютона"""
    global h1, iters, f

    iters = 0
    max_iters: int = 4048
    old_h1: float = 0.0

    makeRealV(flag, interg_flag)
    makeV(flag, interg_flag)
    makePartialV(flag, interg_flag)
    f = funct()

    print(f"\nInitial: h1 = {h1:.6f} m, Φ = {f:.6e}\n")

    while f > eps and iters < max_iters and abs((old_h1 - h1) / (h1 + 1e-12)) > eps:
        old_h1 = h1

        # Формирование СЛАУ: (A + alpha) * delta_h = b
        a: float = alpha  # Регуляризация
        b: float = 0.0

        for i in range(receivers_count):
            a += weights[i] * partial_v[i] ** 2
            b -= weights[i] * partial_v[i] * (v[i] - real_v[i])

        delta_h: float = b / a
        h1 += delta_h

        if h1 <= 0:
            if flag == 'M':
                h1 = abs(h1)
            elif interg_flag == 'S' or interg_flag == 'SIM':
                h1 = 1e-6
            print(f"Warning: h1 became negative, set to {h1} m")

        iters += 1

        makeV(flag, interg_flag)
        makePartialV(flag, interg_flag)
        f = funct()

        # if iters % 10 == 0 or f < eps:
        print(f"Iter {iters:4d}: h1 = {h1:12.9f} m, Φ = {f:14.6e}, Δh = {delta_h:14.6e}, h1 - h_real = {(h1-real_h1):12.9f}, (h1 - h_real)/h_real = {(h1-real_h1)/real_h1:12.9f}")

    print(f"\nКоличество итераций для текущего метода: {iters}")

def compute_condition_number() -> float:
    """Вычисление числа обусловленности (приближённо)"""
    makePartialV()

    # A = sum(w_i * (dV/dh)^2)
    a: float = alpha
    for i in range(receivers_count):
        a += weights[i] * partial_v[i] ** 2

    # Минимальный элемент (только alpha)
    a_min: float = alpha

    condition: float = a / a_min
    return condition

def main() -> None:
    global h1, noise_flag
    f: float = h1
    print("=" * 70)
    print("Two-layer model inverse problem (without noise, with weights, Hankel integral)")
    print("=" * 70)
    init()
    print(f"\nModel parameters:")
    print(f"  σ₁ = {sigma1} S/m (layer 1)")
    print(f"  σ₂ = {sigma2} S/m (layer 2)")
    print(f"  Real h₁ = {real_h1} m")
    print(f"  Initial h₁ = {h1} m")
    print(f"  Source: A({source[0]}, 0, 0), B({source[1]}, 0, 0)")
    print(f"  Current I = {I} A")
    iteration_history: List[int] = []
    while True:
        print('\nВключить шум (±3%)? (y/n): ')
        noise_input = input().strip().lower()
        if noise_input == 'y':
            print("Шум был успешно добавлен")
        noise_flag = noise_input == 'y'
        print('Выберите тип флага:')
        print('1) H - Интеграл Ханкеля')
        print('2) M - Метод зеркальных отображений')
        print('3) Q - Выход')
        flag = input('Введите флаг (H, M or Q): ').strip().upper()
        if flag.lower() == 'q':
            print("Выход из программы.")
            break
        elif flag == 'H':
            print('Выберите метод интегрирования:')
            print('1) S - Интеграл через scipy')
            print('2) SIM - Метод зеркальных отображений')
            interg_flag = input('Введите метод интегрирования (1-2): ').strip()
            if interg_flag == '1':
                interg_flag = 'S'
            elif interg_flag == '2':
                interg_flag = 'SIM'
            else:
                print("Неверный выбор метода, используем scipy по умолчанию.")
                interg_flag = 'S'
        elif flag == 'M':
            interg_flag = 'S'
        else:
            print("Неверный флаг, попробуйте снова.")
            continue
        h1 = f
        solve(flag, interg_flag)
        iteration_history.append(iters)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nFound h₁ = {h1:.12f} m")
    print(f"Real h₁  = {real_h1:.12f} m")
    print(f"Error    = {h1 - real_h1:.12e} m")
    print(f"Rel. err = {(h1 - real_h1) / real_h1 * 100:.10f} %")
    print(f"\nIterations: {iters}")
    print(f"Final Φ:    {f:.6e}")
    print(f"Полная история итераций: {iteration_history}")

    # Число обусловленности
    cond: float = compute_condition_number()
    print(f"Condition number (approx): {cond:.2e}")

    # Таблица невязок
    print(f"\n{'i':^5} {'V_i (calc)':^15} {'V_i (true)':^15} {'Error':^15} {'Weighted err²':^15}")
    print("-" * 75)
    for i in range(3):
        err: float = v[i] - real_v[i]
        w_err2: float = weights[i] * err * err
        print(f"{i + 1:^5} {v[i]:^15.6e} {real_v[i]:^15.6e} {err:^15.6e} {w_err2:^15.6e}")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()