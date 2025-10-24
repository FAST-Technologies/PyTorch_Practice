import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


class SimpleGeneticAlgorithm:
    """Простой генетический алгоритм для решения обратной задачи"""

    def __init__(self,
                 N: int = 5,  # размер генотипа
                 M: int = 3,  # размер фенотипа
                 population_size: int = 50,
                 mutation_rate: float = 0.3,
                 generations: int = 100):
        """
        Инициализация ПГА

        Args:
            N: размер генотипа (количество генов)
            M: размер фенотипа
            population_size: размер популяции
            mutation_rate: вероятность мутации каждого гена
            generations: количество поколений
        """
        self.N = N
        self.M = M
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations

        self.transform_matrix = np.random.randn(M, N)

        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_individual': None
        }

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
            значение функционала (минимизируем)
        """
        phenotype = self.genotype_to_phenotype(genotype)
        return np.sqrt(np.sum(phenotype ** 2))

    def create_random_individual(self) -> np.ndarray:
        """Создание случайной особи"""
        return np.random.uniform(-5, 5, self.N)

    def initialize_population(self) -> List[np.ndarray]:
        """Инициализация начальной популяции"""
        return [self.create_random_individual()
                for _ in range(self.population_size)]

    def mutate(self, genotype: np.ndarray) -> np.ndarray:
        """
        Мутация генотипа (±10% от текущего значения)

        Args:
            genotype: исходный генотип

        Returns:
            мутированный генотип
        """
        mutated = genotype.copy()
        for i in range(len(mutated)):
            if np.random.random() < self.mutation_rate:
                # Изменение на ±10%
                change = mutated[i] * np.random.uniform(-0.1, 0.1)
                mutated[i] += change
        return mutated

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Одноточечный кроссовер

        Args:
            parent1, parent2: родительские генотипы

        Returns:
            потомок
        """
        point = np.random.randint(1, self.N)
        child = np.concatenate([parent1[:point], parent2[point:]])
        return child

    def tournament_selection(self, population: List[np.ndarray],
                             fitness_values: List[float],
                             tournament_size: int = 3) -> np.ndarray:
        """
        Турнирная селекция

        Args:
            population: текущая популяция
            fitness_values: значения fitness для каждой особи
            tournament_size: размер турнира

        Returns:
            выбранная особь
        """
        indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_values[i] for i in indices]
        winner_idx = indices[np.argmin(tournament_fitness)]
        return population[winner_idx]

    def evolve_generation(self, population: List[np.ndarray]) -> List[np.ndarray]:
        """
        Эволюция одного поколения

        Args:
            population: текущая популяция

        Returns:
            новая популяция
        """
        # Вычисляем fitness для всех особей
        fitness_values = [self.calculate_fitness(ind) for ind in population]

        # Сортируем для элитизма
        sorted_indices = np.argsort(fitness_values)

        # Новая популяция
        new_population = []

        # Элитизм - сохраняем лучшую особь
        new_population.append(population[sorted_indices[0]].copy())

        # Создаем остальную популяцию
        while len(new_population) < self.population_size:
            # Селекция родителей
            parent1 = self.tournament_selection(population, fitness_values)
            parent2 = self.tournament_selection(population, fitness_values)

            # Кроссовер
            child = self.crossover(parent1, parent2)

            # Мутация
            child = self.mutate(child)

            new_population.append(child)

        return new_population

    def run(self, verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        Запуск генетического алгоритма

        Args:
            verbose: выводить ли информацию о процессе

        Returns:
            лучший генотип и его fitness
        """
        # Инициализация популяции
        population = self.initialize_population()

        # Эволюция
        for generation in range(self.generations):
            # Вычисляем fitness
            fitness_values = [self.calculate_fitness(ind) for ind in population]

            # Находим лучшую особь
            best_idx = np.argmin(fitness_values)
            best_fitness = fitness_values[best_idx]
            avg_fitness = np.mean(fitness_values)

            # Сохраняем в историю
            self.history['best_fitness'].append(best_fitness)
            self.history['avg_fitness'].append(avg_fitness)

            if verbose and (generation % 10 == 0 or generation == self.generations - 1):
                print(f"Поколение {generation:3d}: "
                      f"Best = {best_fitness:.6f}, "
                      f"Avg = {avg_fitness:.6f}")

            # Эволюция
            if generation < self.generations - 1:
                population = self.evolve_generation(population)

        # Финальный лучший результат
        fitness_values = [self.calculate_fitness(ind) for ind in population]
        best_idx = np.argmin(fitness_values)
        self.history['best_individual'] = population[best_idx]

        return population[best_idx], fitness_values[best_idx]

    def plot_results(self):
        """Визуализация результатов"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Результаты работы простого генетического алгоритма',
                     fontsize=16, fontweight='bold')

        # График эволюции fitness
        ax1 = axes[0, 0]
        generations = range(len(self.history['best_fitness']))
        ax1.plot(generations, self.history['best_fitness'],
                 label='Лучший', linewidth=2, color='#4f46e5')
        ax1.plot(generations, self.history['avg_fitness'],
                 label='Средний', linewidth=2, color='#06b6d4', alpha=0.7)
        ax1.set_xlabel('Поколение')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Эволюция функционала')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Лучший генотип
        ax2 = axes[0, 1]
        best_genotype = self.history['best_individual']
        x_pos = np.arange(len(best_genotype))
        colors = ['#4f46e5' if x >= 0 else '#ef4444' for x in best_genotype]
        ax2.bar(x_pos, best_genotype, color=colors, alpha=0.7)
        ax2.set_xlabel('Индекс гена')
        ax2.set_ylabel('Значение')
        ax2.set_title(f'Лучший генотип (N={self.N})')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3, axis='y')

        # Фенотип
        ax3 = axes[1, 0]
        best_phenotype = self.genotype_to_phenotype(best_genotype)
        x_pos = np.arange(len(best_phenotype))
        colors = ['#10b981' if x >= 0 else '#f59e0b' for x in best_phenotype]
        ax3.bar(x_pos, best_phenotype, color=colors, alpha=0.7)
        ax3.set_xlabel('Индекс фенотипа')
        ax3.set_ylabel('Значение')
        ax3.set_title(f'Фенотип (M={self.M})')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.grid(True, alpha=0.3, axis='y')

        # Информация
        ax4 = axes[1, 1]
        ax4.axis('off')

        info_text = f"""
        Параметры алгоритма:
        ──────────────────────
        Размер генотипа (N): {self.N}
        Размер фенотипа (M): {self.M}
        Размер популяции: {self.population_size}
        Вероятность мутации: {self.mutation_rate}
        Число поколений: {self.generations}

        Результаты:
        ──────────────────────
        Финальный fitness: {self.history['best_fitness'][-1]:.6f}

        Лучший генотип:
        {np.array2string(best_genotype, precision=3, suppress_small=True)}

        Фенотип:
        {np.array2string(best_phenotype, precision=3, suppress_small=True)}

        Функционал: √(Σ фенотипов²)
        Мутация: ±10% от значения
        Селекция: турнирная с элитизмом
        Кроссовер: одноточечный
        """

        ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes,
                 fontsize=10, verticalalignment='top',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        plt.show()


def main():
    """Основная функция для запуска алгоритма"""
    print("=" * 60)
    print("Простой генетический алгоритм (ПГА)")
    print("Решение обратной задачи")
    print("=" * 60)
    print()

    # Создаем и запускаем алгоритм
    ga = SimpleGeneticAlgorithm(
        N=5,  # размер генотипа
        M=3,  # размер фенотипа
        population_size=50,  # размер популяции
        mutation_rate=0.3,  # вероятность мутации
        generations=100  # число поколений
    )

    print("Начинаем эволюцию...\n")
    best_genotype, best_fitness = ga.run(verbose=True)

    print("\n" + "=" * 60)
    print("Эволюция завершена!")
    print("=" * 60)
    print(f"\nЛучший fitness: {best_fitness:.6f}")
    print(f"Лучший генотип: {best_genotype}")
    print(f"Фенотип: {ga.genotype_to_phenotype(best_genotype)}")

    # Визуализация результатов
    ga.plot_results()


if __name__ == "__main__":
    main()