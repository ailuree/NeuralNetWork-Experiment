import numpy as np
import matplotlib.pyplot as plt
from time import time
from typing import Tuple, List

# 目标函数
def fitness_function(x: np.ndarray) -> np.ndarray:
    return x + 10 * np.sin(5 * x) + 7 * np.sin(4 * x)

class GeneticAlgorithm:
    def __init__(self, 
                 pop_size: int = 100,          # 增大种群规模
                 chromosome_length: int = 22,   # 增加编码精度
                 generations: int = 200,        # 增加迭代次数
                 mutation_rate: float = 0.01,
                 crossover_rate: float = 0.8,
                 elite_size: int = 2,           # 精英保留数量
                 convergence_threshold: float = 1e-6):  # 收敛阈值
        self.pop_size = pop_size
        self.chromosome_length = chromosome_length
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.convergence_threshold = convergence_threshold
        
        # 记录运行统计信息
        self.convergence_generation = None
        self.execution_time = None
        
    def initialize_population(self) -> np.ndarray:
        return np.random.randint(0, 2, size=(self.pop_size, self.chromosome_length))
    
    def decode_chromosomes(self, population: np.ndarray) -> np.ndarray:
        decimal_values = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            binary_str = ''.join(map(str, population[i]))
            decimal_values[i] = int(binary_str, 2)
        
        x = decimal_values * 10 / (2**self.chromosome_length - 1)
        return x
    
    def calculate_fitness(self, x: np.ndarray) -> np.ndarray:
        return fitness_function(x)
    
    def select_parents(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        # 保留精英个体
        elite_indices = np.argsort(fitness)[-self.elite_size:]
        elite_population = population[elite_indices]
        
        # 对剩余位置进行轮盘赌选择
        fitness = fitness - np.min(fitness)
        if np.sum(fitness) == 0:
            probs = np.ones(self.pop_size) / self.pop_size
        else:
            probs = fitness / np.sum(fitness)
        
        selected_indices = np.random.choice(
            self.pop_size, 
            size=self.pop_size - self.elite_size, 
            p=probs
        )
        
        # 合并精英个体和选择的个体
        return np.vstack((elite_population, population[selected_indices]))
    
    def crossover(self, parents: np.ndarray) -> np.ndarray:
        offspring = parents.copy()
        # 保护精英个体不参与交叉
        for i in range(self.elite_size, self.pop_size-1, 2):
            if np.random.random() < self.crossover_rate:
                # 使用两点交叉
                points = sorted(np.random.choice(self.chromosome_length-1, 2, replace=False) + 1)
                for j in range(points[0], points[1]):
                    offspring[i,j], offspring[i+1,j] = offspring[i+1,j], offspring[i,j]
        return offspring
    
    def mutate(self, population: np.ndarray) -> np.ndarray:
        # 保护精英个体不参与变异
        mutation_mask = np.random.random(population[self.elite_size:].shape) < self.mutation_rate
        population[self.elite_size:][mutation_mask] = 1 - population[self.elite_size:][mutation_mask]
        return population
    
    def check_convergence(self, fitness_history: List[float], window: int = 20) -> bool:
        """检查是否收敛"""
        if len(fitness_history) < window:
            return False
        recent_fitness = fitness_history[-window:]
        return np.std(recent_fitness) < self.convergence_threshold
    
    def run(self) -> Tuple[List[float], List[float]]:
        start_time = time()
        population = self.initialize_population()
        best_fitness_history = []
        best_x_history = []
        
        for generation in range(self.generations):
            x = self.decode_chromosomes(population)
            fitness = self.calculate_fitness(x)
            
            best_idx = np.argmax(fitness)
            best_fitness_history.append(fitness[best_idx])
            best_x_history.append(x[best_idx])
            
            # 检查收敛性
            if self.check_convergence(best_fitness_history):
                self.convergence_generation = generation
                break
                
            parents = self.select_parents(population, fitness)
            offspring = self.crossover(parents)
            population = self.mutate(offspring)
        
        self.execution_time = time() - start_time
        return best_fitness_history, best_x_history
    
    def get_statistics(self) -> dict:
        """返回算法运行统计信息"""
        return {
            "收敛代数": self.convergence_generation,
            "运行时间": f"{self.execution_time:.4f}秒"
        }

def evaluate_algorithm(n_runs: int = 10) -> None:
    """多次运行算法并进行统计分析"""
    best_solutions = []
    best_fitness_values = []
    convergence_gens = []
    execution_times = []
    
    for i in range(n_runs):
        ga = GeneticAlgorithm()
        best_fitness_history, best_x_history = ga.run()
        stats = ga.get_statistics()
        
        best_idx = np.argmax(best_fitness_history)
        best_solutions.append(best_x_history[best_idx])
        best_fitness_values.append(best_fitness_history[best_idx])
        convergence_gens.append(stats["收敛代数"])
        execution_times.append(float(stats["运行时间"].replace("秒", "")))
    
    print("\n=== 算法性能统计（{}次运行） ===".format(n_runs))
    print(f"最优解平均值: {np.mean(best_solutions):.4f} ± {np.std(best_solutions):.4f}")
    print(f"最优适应度平均值: {np.mean(best_fitness_values):.4f} ± {np.std(best_fitness_values):.4f}")
    print(f"平均收敛代数: {np.mean(convergence_gens):.1f} ± {np.std(convergence_gens):.1f}")
    print(f"平均运行时间: {np.mean(execution_times):.4f} ± {np.std(execution_times):.4f}秒")
    
    return np.mean(best_solutions), np.mean(best_fitness_values)

def plot_results(ga: GeneticAlgorithm, best_fitness_history: List[float], 
                best_x_history: List[float]) -> None:
    """绘制结果图表"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    best_generation = np.argmax(best_fitness_history)
    best_x = best_x_history[best_generation]
    best_fitness = best_fitness_history[best_generation]
    
    plt.figure(figsize=(15, 5))
    
    # 绘制优化过程
    plt.subplot(1, 3, 1)
    plt.plot(best_fitness_history)
    plt.title('优化过程')
    plt.xlabel('代数')
    plt.ylabel('最佳适应度')
    if ga.convergence_generation:
        plt.axvline(x=ga.convergence_generation, color='r', linestyle='--', 
                   label=f'收敛于第{ga.convergence_generation}代')
        plt.legend()
    
    # 绘制目标函数和最优解
    plt.subplot(1, 3, 2)
    x = np.linspace(0, 10, 1000)
    y = fitness_function(x)
    plt.plot(x, y, 'b-', label='目标函数')
    plt.plot(best_x, best_fitness, 'r*', label='最优解')
    plt.title('目标函数和最优解')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    
    # 绘制局部放大图
    plt.subplot(1, 3, 3)
    x_range = 0.5
    x_local = np.linspace(max(0, best_x-x_range), min(10, best_x+x_range), 1000)
    y_local = fitness_function(x_local)
    plt.plot(x_local, y_local, 'b-', label='目标函数')
    plt.plot(best_x, best_fitness, 'r*', label='最优解')
    plt.title('最优解局部放大图')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # 首先进行多次运行的统计分析
    best_x, best_fitness = evaluate_algorithm(n_runs=10)
    
    # 然后展示一次详细的运行结果
    ga = GeneticAlgorithm()
    best_fitness_history, best_x_history = ga.run()
    stats = ga.get_statistics()
    
    print("\n=== 单次运行结果 ===")
    print(f"最优解 x = {best_x_history[np.argmax(best_fitness_history)]:.4f}")
    print(f"最大值 f(x) = {max(best_fitness_history):.4f}")
    print(f"收敛代数: {stats['收敛代数']}")
    print(f"运行时间: {stats['运行时间']}")
    
    # 绘制结果
    plot_results(ga, best_fitness_history, best_x_history)

if __name__ == "__main__":
    main() 