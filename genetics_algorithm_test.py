import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import time
import json
from datetime import datetime


class ImprovedGeneticAlgorithm:
    """
    Простой генетический алгоритм для решения обратной задачи
    Основан на структуре из блок-схемы и C++ примера
    """

    def __init__(self,
                 N: int = 5,
                 M: int = 3,
                 population_size: int = 100,
                 mutation_rate: float = 0.2,
                 max_generations: int = 100,
                 target_fitness: float = 0.01,
                 max_parents: int = 5,
                 adaptive_mutation: bool = True,
                 elitism_count: int = 2):
        """
        Инициализация улучшенного ПГА

        Args:
            N: размер генотипа (количество генов)
            M: размер фенотипа
            population_size: размер популяции
            mutation_rate: начальная вероятность мутации (каждого гена)
            max_generations: максимальное число поколений (MaxP)
            target_fitness: целевое значение функционала (Eps)
            max_parents: число лучших особей для размножения
            adaptive_mutation: использовать ли адаптивную мутацию
            elitism_count: число элитных особей
        """
        self.N = N
        self.M = M
        self.population_size = population_size
        self.population_size_temp = population_size * 2  # Временная популяция
        self.mutation_rate = mutation_rate
        self.initial_mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.target_fitness = target_fitness
        self.max_parents = max_parents
        self.adaptive_mutation = adaptive_mutation
        self.elitism_count = elitism_count

        # Матрица преобразования генотип -> фенотип
        np.random.seed(int(time.time()) % (2 ** 32 - 1))
        self.transform_matrix = np.random.randn(M, N)

        # История эволюции
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'mutation_rate': [],
            'best_individual': None,
            'convergence_generation': None,
            'total_time': 0
        }

        # Для логирования
        self.log_data = []

    def genotype_to_phenotype(self, genotype: np.ndarray) -> np.ndarray:
        """
        Преобразование генотипа в фенотип

        Args:
            genotype: массив из N чисел

        Returns:
            phenotype: массив из M чисел
        """
        return self.transform_matrix @ genotype

    def calculate_fitness(self, genotype: np.ndarray) -> float:
        """
        Вычисление функционала: sqrt(sum(phenotype_i^2))

        Args:
            genotype: генотип особи

        Returns:
            значение функционала F_p_best (минимизируем)
        """
        phenotype = self.genotype_to_phenotype(genotype)
        return np.sqrt(np.sum(phenotype ** 2))

    def create_random_individual(self) -> np.ndarray:
        """Создание случайной особи"""
        return np.random.uniform(-5, 5, self.N)

    def initialize_population(self) -> np.ndarray:
        """
        Генерация начального поколения (p=0)
        Returns: популяция shape (population_size, N)
        """
        return np.array([self.create_random_individual()
                         for _ in range(self.population_size)])

    def mutate(self, genotype: np.ndarray) -> np.ndarray:
        """
        Мутация генотипа (±10% от текущего значения)
        Вероятность мутации может адаптироваться
        Args:
            genotype: исходный генотип

        Returns:
            мутированный генотип
        """
        mutated = genotype.copy()
        mutation_mask = np.random.random(len(mutated)) < self.mutation_rate

        for i in np.where(mutation_mask)[0]:
            change = mutated[i] * np.random.uniform(-0.1, 0.1)
            mutated[i] += change

        return mutated

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Одноточечный кроссинговер
        Args:
            parent1, parent2: родительские генотипы

        Returns:
            потомок
        Note: Аналогично GetNewChild из C++ кода
        """
        crossover_point = np.random.randint(1, self.N)
        child = np.concatenate([parent1[:crossover_point],
                                parent2[crossover_point:]])
        return child

    def select_mother(self, population_size: int) -> int:
        """
        Выбор второго родителя
        Случайный выбор из популяции
        Args:
            population_size: текущий размер популяции

        Returns:
            мать
        Note: Аналог функции GetMother
        """
        return np.random.randint(0, population_size)

    def create_new_generation(self, population: np.ndarray,
                              fitness_values: np.ndarray) -> np.ndarray:
        """
        Генерация нового поколения (кроссинговер, мутация)
        Args:
            population: текущая популяция
            fitness_values: текущие значения функционала

        Returns:
            новая популяция
        Note:
            Аналогично CreateNewPopulation из C++
            Использует max_parents лучших особей как отцов
        """
        # Сортируем по fitness для элитизма
        sorted_indices = np.argsort(fitness_values)

        # Новая временная популяция (в 2 раза больше)
        temp_population = []

        # Количество детей на каждого родителя
        children_per_parent = self.population_size_temp // self.max_parents

        # Генерируем детей от лучших родителей (селекция родителей - по отцу и матери)
        for i in range(self.max_parents):
            father_idx = sorted_indices[i]
            father = population[father_idx]

            for j in range(children_per_parent):
                # Выбираем случайную мать
                mother_idx = self.select_mother(self.population_size)
                mother = population[mother_idx]

                # Кроссовер
                child = self.crossover(father, mother)

                # Мутация
                child = self.mutate(child)

                temp_population.append(child)

        # Дополняем до нужного размера если нужно
        while len(temp_population) < self.population_size_temp:
            parent1 = population[np.random.randint(0, self.population_size)]
            parent2 = population[np.random.randint(0, self.population_size)]
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            temp_population.append(child)

        return np.array(temp_population[:self.population_size_temp])

    def selection(self, temp_population: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Селекция - выбор лучших особей
        Аналогично Selection из C++

        Returns:
            новая популяция, лучший fitness, средний fitness
        """
        # Вычисляем fitness для всех особей
        fitness_values = np.array([self.calculate_fitness(ind)
                                   for ind in temp_population])

        # Сортируем индексы по fitness
        sorted_indices = np.argsort(fitness_values)

        # Выбираем лучших
        new_population = temp_population[sorted_indices[:self.population_size]]

        best_fitness = fitness_values[sorted_indices[0]]
        avg_fitness = np.mean(fitness_values[:self.population_size])

        return new_population, best_fitness, avg_fitness

    def adapt_mutation_rate(self, generation: int, best_fitness: float):
        """
        Адаптивная мутация
        Увеличиваем мутацию если застряли, уменьшаем если улучшается
        """
        if not self.adaptive_mutation:
            return

        # Проверяем застой
        if len(self.history['best_fitness']) > 10:
            recent_improvement = (self.history['best_fitness'][-10] -
                                  self.history['best_fitness'][-1])

            if recent_improvement < 1e-6:
                # Застой - увеличиваем мутацию
                self.mutation_rate = min(0.5, self.mutation_rate * 1.1)
            else:
                # Есть улучшение - уменьшаем мутацию
                self.mutation_rate = max(0.05, self.mutation_rate * 0.95)

    def run(self, verbose: bool = True, log_interval: int = 10) -> Tuple[np.ndarray, float]:
        """
        Запуск генетического алгоритма
        Структура согласно блок-схеме

        Args:
            verbose: выводить ли информацию о процессе
            log_interval: интервал логирования

        Returns:
            лучший генотип и его fitness
        """
        start_time = time.time()

        if verbose:
            print("=" * 70)
            print("Улучшенный простой генетический алгоритм (ПГА)")
            print("=" * 70)
            print(f"Параметры: N={self.N}, M={self.M}, "
                  f"Pop={self.population_size}, MaxGen={self.max_generations}")
            print(f"Целевой fitness (Eps): {self.target_fitness}")
            print("=" * 70)

        # p = 0: Генерация начального поколения / популяции
        population = self.initialize_population()

        # Вычисление функционала для начального поколения
        fitness_values = np.array([self.calculate_fitness(ind)
                                   for ind in population])

        # Находим лучшую особь
        best_idx = np.argmin(fitness_values)
        best_fitness = fitness_values[best_idx]
        avg_fitness = np.mean(fitness_values)

        # Сохраняем историю
        self.history['best_fitness'].append(best_fitness)
        self.history['avg_fitness'].append(avg_fitness)
        self.history['mutation_rate'].append(self.mutation_rate)

        if verbose:
            print(f"Поколение   0: Best = {best_fitness:.8f}, "
                  f"Avg = {avg_fitness:.8f}, MutRate = {self.mutation_rate:.3f}")

        # Основной цикл эволюции (p = p + 1)
        for generation in range(1, self.max_generations + 1):
            # Проверка критерия останова: F_p_best <= Eps или p > MaxP
            if best_fitness <= self.target_fitness:
                if verbose:
                    print(f"\n{'=' * 70}")
                    print(f"Достигнут целевой fitness! Останов на поколении {generation - 1}")
                    print(f"{'=' * 70}")
                self.history['convergence_generation'] = generation - 1
                break

            # Генерация нового поколения (кроссинговер, мутация)
            temp_population = self.create_new_generation(population, fitness_values)

            # Селекция
            population, best_fitness, avg_fitness = self.selection(temp_population)

            # Адаптивная мутация
            self.adapt_mutation_rate(generation, best_fitness)

            # Сохраняем историю
            self.history['best_fitness'].append(best_fitness)
            self.history['avg_fitness'].append(avg_fitness)
            self.history['mutation_rate'].append(self.mutation_rate)

            # Вычисляем fitness для логирования
            fitness_values = np.array([self.calculate_fitness(ind)
                                       for ind in population])

            # Логирование
            if verbose and (generation % log_interval == 0 or
                            generation == self.max_generations):
                print(f"Поколение {generation:3d}: Best = {best_fitness:.8f}, "
                      f"Avg = {avg_fitness:.8f}, MutRate = {self.mutation_rate:.3f}")

            self.log_data.append({
                'generation': generation,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'mutation_rate': self.mutation_rate
            })

        # Финальный результат
        fitness_values = np.array([self.calculate_fitness(ind)
                                   for ind in population])
        # Находим лучшую особь
        best_idx = np.argmin(fitness_values)
        best_individual = population[best_idx]
        best_fitness = fitness_values[best_idx]

        self.history['best_individual'] = best_individual
        self.history['total_time'] = time.time() - start_time

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"Эволюция завершена!")
            print(f"Время выполнения: {self.history['total_time']:.2f} сек")
            print(f"Финальный лучший fitness: {best_fitness:.8f}")
            print(f"{'=' * 70}\n")

        return best_individual, best_fitness

    def save_results(self, filename: str = "ga_results.json"):
        """Сохранение результатов в файл"""
        results = {
            'parameters': {
                'N': self.N,
                'M': self.M,
                'population_size': self.population_size,
                'max_generations': self.max_generations,
                'target_fitness': self.target_fitness,
                'initial_mutation_rate': self.initial_mutation_rate
            },
            'best_genotype': self.history['best_individual'].tolist() if self.history[
                                                                             'best_individual'] is not None else None,
            'best_phenotype': self.genotype_to_phenotype(self.history['best_individual']).tolist() if self.history[
                                                                                                          'best_individual'] is not None else None,
            'final_fitness': self.history['best_fitness'][-1] if self.history['best_fitness'] else None,
            'convergence_generation': self.history['convergence_generation'],
            'total_time': self.history['total_time'],
            'timestamp': datetime.now().isoformat()
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Результаты сохранены в {filename}")

    def plot_results(self):
        """Улучшенная визуализация результатов"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        fig.suptitle('Результаты улучшенного генетического алгоритма',
                     fontsize=16, fontweight='bold')

        generations = range(len(self.history['best_fitness']))

        # 1. График эволюции fitness (большой)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(generations, self.history['best_fitness'],
                 label='Лучший (F_p_best)', linewidth=2.5, color='#4f46e5')
        ax1.plot(generations, self.history['avg_fitness'],
                 label='Средний', linewidth=2, color='#06b6d4', alpha=0.7)
        if self.history['convergence_generation'] is not None:
            ax1.axvline(x=self.history['convergence_generation'],
                        color='red', linestyle='--', linewidth=2,
                        label=f'Сходимость (поколение {self.history["convergence_generation"]})')
        ax1.axhline(y=self.target_fitness, color='green',
                    linestyle=':', linewidth=2, label=f'Цель (Eps={self.target_fitness})')
        ax1.set_xlabel('Поколение (p)', fontsize=11)
        ax1.set_ylabel('Fitness (функционал)', fontsize=11)
        ax1.set_title('Эволюция функционала', fontsize=12, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 2. Адаптивная мутация
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(generations, self.history['mutation_rate'],
                 color='#f59e0b', linewidth=2)
        ax2.set_xlabel('Поколение', fontsize=10)
        ax2.set_ylabel('Вероятность мутации', fontsize=10)
        ax2.set_title('Адаптивная мутация', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. Лучший генотип
        ax3 = fig.add_subplot(gs[1, 0])
        best_genotype = self.history['best_individual']
        x_pos = np.arange(len(best_genotype))
        colors = ['#4f46e5' if x >= 0 else '#ef4444' for x in best_genotype]
        ax3.bar(x_pos, best_genotype, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Индекс гена', fontsize=10)
        ax3.set_ylabel('Значение', fontsize=10)
        ax3.set_title(f'Лучший генотип (N={self.N})', fontsize=11, fontweight='bold')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Фенотип
        ax4 = fig.add_subplot(gs[1, 1])
        best_phenotype = self.genotype_to_phenotype(best_genotype)
        x_pos = np.arange(len(best_phenotype))
        colors = ['#10b981' if x >= 0 else '#f59e0b' for x in best_phenotype]
        ax4.bar(x_pos, best_phenotype, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Индекс фенотипа', fontsize=10)
        ax4.set_ylabel('Значение', fontsize=10)
        ax4.set_title(f'Фенотип (M={self.M})', fontsize=11, fontweight='bold')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.grid(True, alpha=0.3, axis='y')

        # 5. Распределение значений генотипа
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.hist(best_genotype, bins=15, color='#8b5cf6', alpha=0.7, edgecolor='black')
        ax5.set_xlabel('Значение гена', fontsize=10)
        ax5.set_ylabel('Частота', fontsize=10)
        ax5.set_title('Распределение генов', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')

        # 6. Информация о результатах
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')

        info_text = f"""
╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║  ПАРАМЕТРЫ АЛГОРИТМА                          │  РЕЗУЛЬТАТЫ                                  ║
╠═══════════════════════════════════════════════╪══════════════════════════════════════════════╣
║  Размер генотипа (N):          {self.N:3d}           │  Финальный fitness:    {self.history['best_fitness'][-1]:12.8f}  ║
║  Размер фенотипа (M):          {self.M:3d}           │  Начальный fitness:    {self.history['best_fitness'][0]:12.8f}  ║
║  Размер популяции:             {self.population_size:3d}           │  Улучшение:            {(1 - self.history['best_fitness'][-1] / self.history['best_fitness'][0]) * 100:8.2f}%     ║
║  Временная популяция:          {self.population_size_temp:3d}           │  Поколений выполнено:  {len(self.history['best_fitness']) - 1:3d}              ║
║  Начальная мутация:            {self.initial_mutation_rate:.2f}            │  Сходимость:           {'Да' if self.history['convergence_generation'] else 'Нет':3s} (пок. {self.history['convergence_generation'] or 'N/A'})     ║
║  Финальная мутация:            {self.history['mutation_rate'][-1]:.3f}          │  Время выполнения:     {self.history['total_time']:8.2f} сек        ║
║  Макс. поколений (MaxP):       {self.max_generations:3d}           │  Целевой fitness (Eps): {self.target_fitness:.6f}           ║
║  Число родителей (MaxParent):  {self.max_parents:3d}           │  Достигнут целевой:    {'Да' if self.history['best_fitness'][-1] <= self.target_fitness else 'Нет':3s}              ║
║  Элитизм:                      {self.elitism_count:3d}           │                                              ║
║  Адаптивная мутация:           {'Да' if self.adaptive_mutation else 'Нет':3s}           │                                              ║
╠═══════════════════════════════════════════════╧══════════════════════════════════════════════╣
║  ЛУЧШИЙ ГЕНОТИП: {np.array2string(best_genotype, precision=3, separator=', ', max_line_width=90):74s} ║
║  ФЕНОТИП:        {np.array2string(best_phenotype, precision=3, separator=', ', max_line_width=90):74s} ║
╠══════════════════════════════════════════════════════════════════════════════════════════════╣
║  Функционал: √(Σ фенотипов²)  │  Мутация: ±10%  │  Селекция: на основе fitness + элитизм     ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝
        """

        ax6.text(0.5, 0.5, info_text, transform=ax6.transAxes,
                 fontsize=9, verticalalignment='center', horizontalalignment='center',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#f0f9ff',
                           alpha=0.8, edgecolor='#3b82f6', linewidth=2))

        plt.savefig('ga_results.png', dpi=300, bbox_inches='tight')
        print("График сохранен в ga_results.png")
        plt.show()


def main():
    """Основная функция с демонстрацией всех улучшений"""
    print("\n" + "=" * 80)
    print("ПРОСТОЙ ГЕНЕТИЧЕСКИЙ АЛГОРИТМ (ПГА)")
    print("=" * 80 + "\n")

    # Создаём и запускаем алгоритм
    N, M = 5, 5
    ga = ImprovedGeneticAlgorithm(
        N=5,  # размер генотипа (число генов в изн коде 50)
        M=5,  # размер фенотипа
        population_size=50,  # размер популяции (в изначальном коде 10000)
        mutation_rate=0.2,  # начальная вероятность мутации
        max_generations=100,  # MaxP - максимум поколений (в изначальном коде 50)
        target_fitness=1e-2,  # Eps - целевой fitness (значение ф-онала для выхода в изн коде 0.1)
    )

    print(" Начинаем эволюцию...\n")
    # ga.transform_matrix = np.zeros((M, N)) - заполнение матрицы нулями
    ga.transform_matrix = np.eye(M) # Заполнение единичной матрицы
    start_time = time.time()
    best_genotype, best_fitness = ga.run(verbose=True, log_interval=10)
    execution_time = time.time() - start_time

    genotype_norm = np.linalg.norm(best_genotype)

    success = best_fitness < 0.1 and genotype_norm < 1.0

    print("\n" + "=" * 80)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print("=" * 80)
    print(f"Лучший fitness:          {best_fitness:.8f}")
    print(f"Лучший генотип:          {np.array2string(best_genotype, precision=4, suppress_small=True)}")
    print(
        f"Соответствующий фенотип: {np.array2string(ga.genotype_to_phenotype(best_genotype), precision=4, suppress_small=True)}")
    print(
        f"\nЭффективность направленных мутаций: {ga.mutation_stats['accepted'] / (ga.mutation_stats['total_attempts'] + 1e-10) * 100:.1f}%")
    print(f"  • Принято улучшающих:  {ga.mutation_stats['accepted']}")
    print(f"  • Отвергнуто ухудшающих: {ga.mutation_stats['rejected']}")
    print(f"Success: {success}")
    print(f"Norm: {genotype_norm}")
    print(f"Execution_time: {execution_time}")
    print("=" * 80 + "\n")

    # Сохранение результатов
    ga.save_results("ga_advanced_results.json")

    # Визуализация
    ga.plot_results()

    print("\nГотово! Проверьте файлы ga_results.json и ga_results.png")


if __name__ == "__main__":
    main()