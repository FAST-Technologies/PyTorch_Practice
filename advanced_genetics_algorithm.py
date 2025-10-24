import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import time
import json
from datetime import datetime

class AdvancedGeneticAlgorithm:
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
                 elitism_count: int = 3,
                 directed_mutation_ratio: float = 0.7,
                 diversity_threshold: float = 0.01) -> None:
        """
        Args:
            N: размер генотипа
            M: размер фенотипа
            population_size: размер популяции
            mutation_rate: вероятность мутации
            max_generations: макс. число поколений
            target_fitness: целевой функционал (Eps)
            max_parents: число лучших родителей
            elitism_count: число элитных особей
            directed_mutation_ratio: доля направленных мутаций (0.7 = 70%)
            diversity_threshold: порог разнообразия
        """
        self.N: int = N
        self.M: int = M
        self.population_size: int = population_size
        self.population_size_temp: int = population_size * 2 # Временная популяция
        self.mutation_rate: float = mutation_rate
        self.initial_mutation_rate: float = mutation_rate
        self.max_generations: int = max_generations
        self.target_fitness: float = target_fitness
        self.max_parents: int = max_parents
        self.elitism_count: int = elitism_count
        self.directed_mutation_ratio: float = directed_mutation_ratio
        self.diversity_threshold: float = diversity_threshold

        # Матрица преобразования генотип -> фенотип
        np.random.seed(int(time.time()) % (2 ** 32 - 1))
        self.transform_matrix = np.random.randn(M, N)

        # История эволюции
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'diversity': [],
            'mutation_rate': [],
            'accepted_mutations': [],
            'rejected_mutations': [],
            'best_individual': None,
            'convergence_generation': None,
            'total_time': 0
        }

        # Статистика мутаций
        self.mutation_stats = {
            'total_attempts': 0,
            'accepted': 0,
            'rejected': 0
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
            значение функционала F_p_best (минимизируем)
        """
        phenotype = self.genotype_to_phenotype(genotype)
        return np.sqrt(np.sum(phenotype ** 2))

    def calculate_diversity(self, population: np.ndarray) -> float:
        """
        Вычисление разнообразия популяции
        Используем стандартное отклонение fitness

        Args:
            population: текущая популяция

        Returns:
            значение квадратичного отклонения
        """
        fitness_values = np.array([self.calculate_fitness(ind) for ind in population])
        return np.std(fitness_values)

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

    def directed_mutation(self,
                          genotype: np.ndarray,
                          current_fitness: float,
                          max_attempts: int = 5) -> Tuple[np.ndarray, bool]:
        """
        Направленная мутация генотипа - принимаем только улучшающие изменения
        (±10% от текущего значения)
        Args:
            genotype: исходный генотип
            current_fitness: текущее значение функционала
            max_attempts: максимальное число попыток

        Returns:
            мутированный генотип в случае успеха
        """
        self.mutation_stats['total_attempts'] += 1

        best_mutant = genotype.copy()
        best_fitness = current_fitness
        found_improvement = False

        for _ in range(max_attempts):
            mutant = genotype.copy()

            gene_idx = np.random.randint(0, self.N)

            change = mutant[gene_idx] * np.random.uniform(-0.1, 0.1)
            mutant[gene_idx] += change

            # Проверяем улучшение
            mutant_fitness = self.calculate_fitness(mutant)

            if mutant_fitness < best_fitness:
                best_mutant = mutant
                best_fitness = mutant_fitness
                found_improvement = True

        if found_improvement:
            self.mutation_stats['accepted'] += 1
        else:
            self.mutation_stats['rejected'] += 1

        return best_mutant, found_improvement

    def random_mutation(self, genotype: np.ndarray) -> np.ndarray:
        """
        Случайная недирективная мутация генотипа (±10% от текущего значения)
        Для поддержания разнообразия популяции
        Args:
            genotype: исходный генотип

        Returns:
            мутированный генотип
        """
        mutated = genotype.copy()

        for i in range(len(mutated)):
            if np.random.random() < self.mutation_rate:
                change = mutated[i] * np.random.uniform(-0.1, 0.1)
                mutated[i] += change

        return mutated

    def hybrid_mutation(self, genotype: np.ndarray,
                        current_fitness: float) -> np.ndarray:
        """
        Гибридная мутация: комбинация направленной и случайной (±10% от текущего значения)

        Args:
            genotype: исходный генотип
            current_fitness: текущее значение функционала

        Returns:
            мутированный генотип

        Note:
            - directed_mutation_ratio часть - направленные мутации
            - остальное - случайные мутации для разнообразия
        """
        if np.random.random() < self.directed_mutation_ratio:
            # Направленная мутация
            mutant, success = self.directed_mutation(genotype, current_fitness)
            if success:
                return mutant
            # Если не удалось улучшить, возвращаем оригинал
            return genotype
        else:
            # Случайная мутация
            return self.random_mutation(genotype)

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

    def proportional_selection(self, population: np.ndarray,
                               fitness_values: np.ndarray) -> np.ndarray:
        """
        Пропорциональная селекция (рулетка)
        Вероятность выбора обратно пропорциональна fitness

        Args:
            population: текущая популяция
            fitness_values: текущие значения функционала

        Returns:
            результат селекции
        """
        # Инвертируем fitness (меньше = лучше)
        # Добавляем малую константу для избежания деления на ноль
        inv_fitness = 1.0 / (fitness_values + 1e-10)

        # Нормализуем в вероятности
        probabilities = inv_fitness / np.sum(inv_fitness)

        # Выбираем индекс согласно вероятностям
        idx = np.random.choice(len(population), p=probabilities)
        return population[idx]

    def tournament_selection(self, population: np.ndarray,
                             fitness_values: np.ndarray,
                             tournament_size: int = 3) -> np.ndarray:
        """
        Турнирная селекция

        Args:
            population: текущая популяция
            fitness_values: текущие значения функционала
            tournament_size: размер выбираемых родителей

        Returns:
            результат селекции
        """
        indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = fitness_values[indices]
        winner_idx = indices[np.argmin(tournament_fitness)]
        return population[winner_idx]

    def create_new_generation(self, population: np.ndarray,
                              fitness_values: np.ndarray) -> np.ndarray:
        """
        Генерация нового поколения (кроссинговер, мутация)
        Args:
            population: текущая популяция
            fitness_values: текущие значения функционала

        Returns:
            новая популяция
        """
        # Сортируем по fitness для элитизма
        sorted_indices = np.argsort(fitness_values)

        # Новая временная популяция (в 2 раза больше)
        temp_population = []

        # ЭЛИТИЗМ - лучшие особи переходят без изменений
        for i in range(self.elitism_count):
            temp_population.append(population[sorted_indices[i]].copy())

        # Генерация потомков от лучших родителей - Количество детей на каждого родителя
        children_per_parent = (self.population_size_temp - self.elitism_count) // self.max_parents

        # Генерируем детей от лучших родителей (селекция родителей - по отцу и матери)
        for i in range(self.max_parents):
            father_idx = sorted_indices[i]
            father = population[father_idx]
            father_fitness = fitness_values[father_idx]

            for j in range(children_per_parent):
                # Выбор матери (пропорциональная или турнирная селекция)
                if np.random.random() < 0.5:
                    mother = self.proportional_selection(population, fitness_values)
                else:
                    mother = self.tournament_selection(population, fitness_values)

                # Кроссовер
                child = self.crossover(father, mother)

                # ГИБРИДНАЯ МУТАЦИЯ
                child_fitness = self.calculate_fitness(child)
                child = self.hybrid_mutation(child, child_fitness)

                temp_population.append(child)

        # Дополняем до нужного размера если нужно
        while len(temp_population) < self.population_size_temp:
            parent1 = self.tournament_selection(population, fitness_values)
            parent2 = self.tournament_selection(population, fitness_values)
            child = self.crossover(parent1, parent2)
            child_fitness = self.calculate_fitness(child)
            child = self.hybrid_mutation(child, child_fitness)
            temp_population.append(child)

        return np.array(temp_population[:self.population_size_temp])

    def selection(self, temp_population: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Селекция - выбор лучших особей
        Аналогично Selection из C++

        Args:
            temp_population: временная популяция

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

    def adapt_mutation_rate(self, diversity: float) -> None:
        """
        Адаптивная мутация на основе разнообразия популяции

        Args:
            diversity: текущее разнообразие
        """
        # Если разнообразие низкое - увеличиваем мутацию
        if diversity < self.diversity_threshold:
            self.mutation_rate = min(0.5, self.mutation_rate * 1.15)
            # Увеличиваем долю случайных мутаций
            self.directed_mutation_ratio = max(0.3, self.directed_mutation_ratio * 0.95)
        else:
            # Разнообразие достаточное - можем быть более направленными
            self.mutation_rate = max(0.05, self.mutation_rate * 0.98)
            self.directed_mutation_ratio = min(0.9, self.directed_mutation_ratio * 1.02)

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
            print("=" * 80)
            print("Улучшенный простой генетический алгоритм (ПГА)")
            print("=" * 80)
            print(f"Параметры: N={self.N}, M={self.M}, Pop={self.population_size}")
            print(f"Направленные мутации: {self.directed_mutation_ratio * 100:.0f}%")
            print(f"Целевой fitness: {self.target_fitness}")
            print("=" * 80)

        # p = 0: Генерация начального поколения / популяции
        population = self.initialize_population()

        # Вычисление функционала для начального поколения
        fitness_values = np.array([self.calculate_fitness(ind) for ind in population])

        # Находим лучшую особь на начальном этапе
        best_idx = np.argmin(fitness_values)
        best_fitness = fitness_values[best_idx]
        avg_fitness = np.mean(fitness_values)
        diversity = self.calculate_diversity(population)

        self.history['best_fitness'].append(best_fitness)
        self.history['avg_fitness'].append(avg_fitness)
        self.history['diversity'].append(diversity)
        self.history['mutation_rate'].append(self.mutation_rate)
        self.history['accepted_mutations'].append(0)
        self.history['rejected_mutations'].append(0)

        if verbose:
            print(f"Пок.   0: Best={best_fitness:.8f}, Avg={avg_fitness:.6f}, "
                  f"Div={diversity:.6f}, Mut={self.mutation_rate:.3f}")

        # Основной цикл эволюции (p = p + 1)
        for generation in range(1, self.max_generations + 1):
            # Проверка критерия останова: F_p_best <= Eps или p > MaxP
            if best_fitness <= self.target_fitness:
                if verbose:
                    print(f"\n{'=' * 80}")
                    print(f"Достигнут целевой fitness на поколении {generation - 1}!")
                    print(f"{'=' * 80}")
                print(f"Пок. {generation:3d}: Best={best_fitness:.8f}, Avg={avg_fitness:.6f}, "
                      f"Div={diversity:.6f}, Mut={self.mutation_rate:.3f}, "
                      f"Dir={self.directed_mutation_ratio:.2f}, +Mut={accepted}, -Mut={rejected}")
                self.history['convergence_generation'] = generation - 1
                break

            # Сохраняем статистику мутаций перед поколением
            mut_before = self.mutation_stats.copy()

            # Генерация нового поколения (кроссинговер, мутация)
            temp_population = self.create_new_generation(population, fitness_values)

            # Селекция
            population, best_fitness, avg_fitness = self.selection(temp_population)

            # Обновляем fitness для новой популяции
            fitness_values = np.array([self.calculate_fitness(ind) for ind in population])

            # Вычисляем разнообразие
            diversity = self.calculate_diversity(population)

            # Адаптивная мутация
            self.adapt_mutation_rate(diversity)

            # Статистика мутаций за это поколение
            accepted = self.mutation_stats['accepted'] - mut_before['accepted']
            rejected = self.mutation_stats['rejected'] - mut_before['rejected']

            # Сохраняем историю
            self.history['best_fitness'].append(best_fitness)
            self.history['avg_fitness'].append(avg_fitness)
            self.history['diversity'].append(diversity)
            self.history['mutation_rate'].append(self.mutation_rate)
            self.history['accepted_mutations'].append(accepted)
            self.history['rejected_mutations'].append(rejected)

            # Логирование
            if verbose and (generation % log_interval == 0 or generation == self.max_generations):
                print(f"Пок. {generation:3d}: Best={best_fitness:.8f}, Avg={avg_fitness:.6f}, "
                      f"Div={diversity:.6f}, Mut={self.mutation_rate:.3f}, "
                      f"Dir={self.directed_mutation_ratio:.2f}, +Mut={accepted}, -Mut={rejected}")

                # Предупреждение о низком разнообразии
                if diversity < self.diversity_threshold:
                    print(f"           Низкое разнообразие! Увеличена случайная мутация")

        # Финальный результат
        best_idx = np.argmin(fitness_values)
        best_individual = population[best_idx]
        best_fitness = fitness_values[best_idx]

        self.history['best_individual'] = best_individual
        self.history['total_time'] = time.time() - start_time

        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Эволюция завершена!")
            print(f"Время выполнения: {self.history['total_time']:.2f} сек")
            print(f"Финальный fitness: {best_fitness:.8f}")
            print(f"Статистика мутаций: принято={self.mutation_stats['accepted']}, "
                  f"отвергнуто={self.mutation_stats['rejected']}, "
                  f"эффективность={self.mutation_stats['accepted'] / (self.mutation_stats['total_attempts'] + 1e-10) * 100:.1f}%")
            print(f"{'=' * 80}\n")

        return best_individual, best_fitness

    def save_results(self, filename: str = "ga_advanced_results.json") -> None:
        """
        Сохранение результатов в файл
        Args:
            filename: название файла
        """
        results = {
            'parameters': {
                'N': self.N,
                'M': self.M,
                'population_size': self.population_size,
                'max_generations': self.max_generations,
                'target_fitness': self.target_fitness,
                'initial_mutation_rate': self.initial_mutation_rate,
                'directed_mutation_ratio': self.directed_mutation_ratio,
                'diversity_threshold': self.diversity_threshold
            },
            'best_genotype': self.history['best_individual'].tolist() if self.history[
                                                                             'best_individual'] is not None else None,
            'best_phenotype': self.genotype_to_phenotype(self.history['best_individual']).tolist() if self.history[
                                                                                                          'best_individual'] is not None else None,
            'final_fitness': self.history['best_fitness'][-1] if self.history['best_fitness'] else None,
            'convergence_generation': self.history['convergence_generation'],
            'total_time': self.history['total_time'],
            'mutation_statistics': self.mutation_stats,
            'timestamp': datetime.now().isoformat()
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Результаты сохранены в {filename}")

    def plot_results(self):
        """Расширенная визуализация"""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

        fig.suptitle('Результаты продвинутого генетического алгоритма',
                     fontsize=16, fontweight='bold')

        generations = range(len(self.history['best_fitness']))

        # 1. Эволюция fitness
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(generations, self.history['best_fitness'],
                 label='Лучший', linewidth=2.5, color='#4f46e5')
        ax1.plot(generations, self.history['avg_fitness'],
                 label='Средний', linewidth=2, color='#06b6d4', alpha=0.7)
        if self.history['convergence_generation']:
            ax1.axvline(x=self.history['convergence_generation'],
                        color='red', linestyle='--', linewidth=2, label='Сходимость')
        ax1.axhline(y=self.target_fitness, color='green',
                    linestyle=':', linewidth=2, label=f'Цель={self.target_fitness}')
        ax1.set_xlabel('Поколение')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Эволюция функционала')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 2. Разнообразие популяции
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(generations, self.history['diversity'],
                 color='#8b5cf6', linewidth=2)
        ax2.axhline(y=self.diversity_threshold, color='red',
                    linestyle='--', label=f'Порог={self.diversity_threshold}')
        ax2.set_xlabel('Поколение')
        ax2.set_ylabel('Разнообразие (std)')
        ax2.set_title('Контроль разнообразия')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Адаптация мутации
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(generations, self.history['mutation_rate'],
                 color='#f59e0b', linewidth=2)
        ax3.set_xlabel('Поколение')
        ax3.set_ylabel('Вероятность мутации')
        ax3.set_title('Адаптивная мутация')
        ax3.grid(True, alpha=0.3)

        # 4. Статистика направленных мутаций
        ax4 = fig.add_subplot(gs[1, 1])
        accepted = self.history['accepted_mutations']
        rejected = self.history['rejected_mutations']
        x = list(generations)
        ax4.bar(x, accepted, label='Принято', color='green', alpha=0.7)
        ax4.bar(x, rejected, bottom=accepted, label='Отвергнуто', color='red', alpha=0.7)
        ax4.set_xlabel('Поколение')
        ax4.set_ylabel('Число мутаций')
        ax4.set_title('Направленные мутации')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        # 5. Эффективность мутаций
        ax5 = fig.add_subplot(gs[1, 2])
        total = np.array(accepted) + np.array(rejected)
        efficiency = np.where(total > 0, np.array(accepted) / total * 100, 0)
        ax5.plot(generations, efficiency, color='#10b981', linewidth=2)
        ax5.set_xlabel('Поколение')
        ax5.set_ylabel('Эффективность (%)')
        ax5.set_title('% принятых мутаций')
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim([0, 100])

        # 6-8. Генотип, фенотип, распределение
        best_genotype = self.history['best_individual']
        best_phenotype = self.genotype_to_phenotype(best_genotype)

        ax6 = fig.add_subplot(gs[2, 0])
        colors = ['#4f46e5' if x >= 0 else '#ef4444' for x in best_genotype]
        ax6.bar(range(len(best_genotype)), best_genotype, color=colors, alpha=0.7, edgecolor='black')
        ax6.set_xlabel('Индекс гена')
        ax6.set_ylabel('Значение')
        ax6.set_title(f'Лучший генотип (N={self.N})')
        ax6.axhline(y=0, color='black', linewidth=0.5)
        ax6.grid(True, alpha=0.3, axis='y')

        ax7 = fig.add_subplot(gs[2, 1])
        colors = ['#10b981' if x >= 0 else '#f59e0b' for x in best_phenotype]
        ax7.bar(range(len(best_phenotype)), best_phenotype, color=colors, alpha=0.7, edgecolor='black')
        ax7.set_xlabel('Индекс фенотипа')
        ax7.set_ylabel('Значение')
        ax7.set_title(f'Фенотип (M={self.M})')
        ax7.axhline(y=0, color='black', linewidth=0.5)
        ax7.grid(True, alpha=0.3, axis='y')

        ax8 = fig.add_subplot(gs[2, 2])
        ax8.hist(best_genotype, bins=15, color='#8b5cf6', alpha=0.7, edgecolor='black')
        ax8.set_xlabel('Значение')
        ax8.set_ylabel('Частота')
        ax8.set_title('Распределение генов')
        ax8.grid(True, alpha=0.3, axis='y')

        # 9. Информация
        ax9 = fig.add_subplot(gs[3, :])
        ax9.axis('off')

        mut_efficiency = self.mutation_stats['accepted'] / (self.mutation_stats['total_attempts'] + 1e-10) * 100

        info = f"""
╔════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║  ПАРАМЕТРЫ АЛГОРИТМА                    │  РЕЗУЛЬТАТЫ                    │  СТАТИСТИКА МУТАЦИЙ         ║
╠═════════════════════════════════════════╪════════════════════════════════╪═════════════════════════════╣
║  Генотип (N):              {self.N:3d}          │  Финальный fitness: {self.history['best_fitness'][-1]:9.6f}  │  Всего попыток:  {self.mutation_stats['total_attempts']:6d}   ║
║  Фенотип (M):              {self.M:3d}          │  Начальный:         {self.history['best_fitness'][0]:9.6f}  │  Принято:        {self.mutation_stats['accepted']:6d}   ║
║  Популяция:                {self.population_size:3d}          │  Улучшение:         {(1 - self.history['best_fitness'][-1] / self.history['best_fitness'][0]) * 100:8.2f}%  │  Отвергнуто:     {self.mutation_stats['rejected']:6d}   ║
║  Макс. поколений:          {self.max_generations:3d}          │  Поколений:         {len(self.history['best_fitness']) - 1:6d}      │  Эффективность:  {mut_efficiency:6.1f}%   ║
║  Направл. мутации:         {self.directed_mutation_ratio * 100:3.0f}%         │  Сходимость:        {'Да' if self.history['convergence_generation'] else 'Нет':7s}     │                             ║
║  Число элиты:              {self.elitism_count:3d}          │  Время:             {self.history['total_time']:8.2f} сек  │  УЛУЧШЕНИЯ:                 ║
║  MaxParent:                {self.max_parents:3d}          │  Целевой (Eps):     {self.target_fitness:9.6f}  │  ✓ Направленные мутации     ║
║  Порог разнообразия:       {self.diversity_threshold:.3f}       │  Достигнут:         {'Да' if self.history['best_fitness'][-1] <= self.target_fitness else 'Нет':7s}     │  ✓ Контроль разнообразия    ║
║                                         │  Фин. разнообразие: {self.history['diversity'][-1]:9.6f}  │  ✓ Гибридная стратегия      ║
╚═════════════════════════════════════════╧════════════════════════════════╧═════════════════════════════╝
        """

        ax9.text(0.5, 0.5, info, transform=ax9.transAxes,
                 fontsize=9, verticalalignment='center', horizontalalignment='center',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#f0f9ff',
                           alpha=0.9, edgecolor='#3b82f6', linewidth=2))

        plt.savefig('ga_advanced_results.png', dpi=300, bbox_inches='tight')
        print("График сохранён в ga_advanced_results.png")
        plt.show()

def main():
    """Основная функция с демонстрацией всех улучшений"""
    print("\n" + "=" * 80)
    print("ПРОСТОЙ ГЕНЕТИЧЕСКИЙ АЛГОРИТМ (ПГА)")
    print("=" * 80 + "\n")

    # Создаём и запускаем алгоритм
    N, M = 10, 100
    ga = AdvancedGeneticAlgorithm(
        N=N,  # размер генотипа
        M=M,  # размер фенотипа
        population_size=100,  # размер популяции
        mutation_rate=0.2,  # начальная вероятность мутации
        max_generations=200,  # макс. поколений
        target_fitness=0.01,  # целевой fitness (Eps)
        max_parents=5,  # число лучших родителей
        elitism_count=3,  # число элитных особей
        directed_mutation_ratio=0.7,  # 70% направленных мутаций
        diversity_threshold=0.01  # порог разнообразия
    )

    print(" Начинаем эволюцию...\n")
    # ga.transform_matrix = np.zeros((M, N)) - заполнение матрицы нулями
    # ga.transform_matrix = np.eye(M) # Заполнение единичной матрицы
    # np.random.seed(111)
    # ga.transform_matrix = np.random.randn(M, N)
    # mask = np.random.random((M, N)) < 0.9
    # ga.transform_matrix[mask] = 0
    # ga.transform_matrix = [[1/(j + i - 1) for j in range(1, N + 1, 1)] for i in range(1, M + 1, 1)]
    # for i in range(M):
    #     for j in range(N):
    #         print(f'{ga.transform_matrix[i][j]} ')
    #     print('\n')

    sparsity =  np.sum(ga.transform_matrix == 0) / (M * N) * 100

    start_time = time.time()
    # best_genotype, best_fitness = ga.run(verbose=True, log_interval=10)
    best_genotype, best_fitness = ga.run(verbose=True)
    execution_time = time.time() - start_time

    genotype_norm = np.linalg.norm(best_genotype)

    success = best_fitness < 0.5

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
    print(f"Sparsity: {sparsity}")
    print(f"Execution_time: {execution_time}")
    print("=" * 80 + "\n")

    # Сохранение результатов
    ga.save_results("ga_advanced_results.json")

    # Визуализация
    ga.plot_results()

    print("\nГотово! Проверьте файлы ga_results.json и ga_results.png")

    return ga


def compare_algorithms():
    """
    Сравнение базового и продвинутого алгоритмов
    """
    print("\n" + "=" * 80)
    print("🔬 СРАВНЕНИЕ АЛГОРИТМОВ")
    print("=" * 80)

    N, M = 6, 4
    np.random.seed(42)
    test_matrix = np.random.randn(M, N)

    # 1. Базовый алгоритм (только случайные мутации)
    print("\n1️⃣  Запуск БАЗОВОГО алгоритма (случайные мутации)...")
    ga_basic = AdvancedGeneticAlgorithm(
        N=N, M=M,
        population_size=80,
        max_generations=150,
        target_fitness=0.05,
        directed_mutation_ratio=0.0,  # Только случайные мутации
        diversity_threshold=0.01
    )
    ga_basic.transform_matrix = test_matrix.copy()

    start = time.time()
    _, fitness_basic = ga_basic.run(verbose=False)
    time_basic = time.time() - start
    gen_basic = len(ga_basic.history['best_fitness']) - 1

    # 2. Продвинутый алгоритм (70% направленные мутации)
    print("\n2️⃣  Запуск ПРОДВИНУТОГО алгоритма (70% направленных мутаций)...")
    ga_advanced = AdvancedGeneticAlgorithm(
        N=N, M=M,
        population_size=80,
        max_generations=150,
        target_fitness=0.05,
        directed_mutation_ratio=0.7,  # 70% направленных
        diversity_threshold=0.01
    )
    ga_advanced.transform_matrix = test_matrix.copy()

    start = time.time()
    _, fitness_advanced = ga_advanced.run(verbose=False)
    time_advanced = time.time() - start
    gen_advanced = len(ga_advanced.history['best_fitness']) - 1

    # Сравнение
    print("\n" + "=" * 80)
    print("📊 РЕЗУЛЬТАТЫ СРАВНЕНИЯ:")
    print("=" * 80)
    print(f"\n{'Метрика':<30} {'Базовый':<20} {'Продвинутый':<20} {'Улучшение':<15}")
    print("-" * 85)
    print(
        f"{'Финальный fitness':<30} {fitness_basic:<20.8f} {fitness_advanced:<20.8f} {(fitness_basic - fitness_advanced) / fitness_basic * 100:>13.1f}%")
    print(
        f"{'Число поколений':<30} {gen_basic:<20d} {gen_advanced:<20d} {(gen_basic - gen_advanced) / gen_basic * 100:>13.1f}%")
    print(
        f"{'Время выполнения (сек)':<30} {time_basic:<20.2f} {time_advanced:<20.2f} {(time_basic - time_advanced) / time_basic * 100:>13.1f}%")

    eff_basic = ga_basic.mutation_stats['accepted'] / (ga_basic.mutation_stats['total_attempts'] + 1e-10) * 100
    eff_advanced = ga_advanced.mutation_stats['accepted'] / (ga_advanced.mutation_stats['total_attempts'] + 1e-10) * 100
    print(
        f"{'Эффективность мутаций (%)':<30} {eff_basic:<20.1f} {eff_advanced:<20.1f} {eff_advanced - eff_basic:>13.1f}pp")
    print("=" * 80)

    if fitness_advanced < fitness_basic:
        print("\n✅ Продвинутый алгоритм показал ЛУЧШИЕ результаты!")
    else:
        print("\n⚠️  Базовый алгоритм показал лучшие результаты в этом тесте")

    # Визуализация сравнения
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # График 1: Эволюция fitness
    ax1 = axes[0]
    ax1.plot(ga_basic.history['best_fitness'], label='Базовый', linewidth=2)
    ax1.plot(ga_advanced.history['best_fitness'], label='Продвинутый', linewidth=2)
    ax1.set_xlabel('Поколение')
    ax1.set_ylabel('Best Fitness')
    ax1.set_title('Сравнение сходимости')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # График 2: Разнообразие
    ax2 = axes[1]
    ax2.plot(ga_basic.history['diversity'], label='Базовый', linewidth=2)
    ax2.plot(ga_advanced.history['diversity'], label='Продвинутый', linewidth=2)
    ax2.set_xlabel('Поколение')
    ax2.set_ylabel('Разнообразие')
    ax2.set_title('Контроль разнообразия')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # График 3: Столбчатая диаграмма метрик
    ax3 = axes[2]
    metrics = ['Fitness\n(ниже лучше)', 'Поколений\n(меньше лучше)', 'Эффект.\nмутаций']
    basic_vals = [fitness_basic / fitness_advanced, gen_basic / gen_advanced, eff_basic / 100]
    advanced_vals = [1.0, 1.0, eff_advanced / 100]

    x = np.arange(len(metrics))
    width = 0.35
    ax3.bar(x - width / 2, basic_vals, width, label='Базовый', alpha=0.7)
    ax3.bar(x + width / 2, advanced_vals, width, label='Продвинутый', alpha=0.7)
    ax3.set_ylabel('Относительное значение')
    ax3.set_title('Сравнение метрик')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, fontsize=9)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=1.0, color='black', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
    print("\n📊 График сравнения сохранён в algorithm_comparison.png")
    plt.show()


if __name__ == "__main__":
    # Демонстрация продвинутого алгоритма
    ga = main()

    # Сравнение с базовым
    print("\n" + "=" * 80)
    response = input("Запустить сравнение с базовым алгоритмом? (y/n): ")
    if response.lower() == 'y':
        compare_algorithms()

    print("\n🎉 Все готово!")