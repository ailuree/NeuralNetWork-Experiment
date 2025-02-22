import numpy as np
import matplotlib.pyplot as plt
from time import time
from typing import Tuple, List

# 目标函数（Sphere Function）
def fitness_function(x: np.ndarray) -> float:
    """计算适应度值
    x: shape为(n,)的numpy数组，表示n维空间中的一个点
    返回: f(x) = 20 + sum(xi^2)
    """
    return 20 + np.sum(x**2)

class GeneticAlgorithm:
    def __init__(self,
                 dim: int = 10,              # 问题维度
                 pop_size: int = 200,        # 更大的种群规模
                 chromosome_length: int = 20, # 每个变量的编码长度
                 generations: int = 500,      # 更多的迭代次数
                 mutation_rate: float = 0.01,
                 crossover_rate: float = 0.8,
                 elite_size: int = 4,
                 convergence_threshold: float = 1e-6):
        self.dim = dim
        self.pop_size = pop_size
        self.chromosome_length = chromosome_length
        self.total_length = chromosome_length * dim  # 总染色体长度
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.convergence_threshold = convergence_threshold
        self.bounds = [-20, 20]  # 每个维度的取值范围
        
        self.convergence_generation = None
        self.execution_time = None
        
    def initialize_population(self) -> np.ndarray:
        """初始化种群，返回shape为(pop_size, total_length)的二进制数组"""
        return np.random.randint(0, 2, size=(self.pop_size, self.total_length))
    
    def decode_chromosomes(self, population: np.ndarray) -> np.ndarray:
        """将二进制染色体解码为实数向量"""
        decoded = np.zeros((self.pop_size, self.dim))
        for i in range(self.pop_size):
            for j in range(self.dim):
                # 提取每个维度对应的二进制片段
                start = j * self.chromosome_length
                end = (j + 1) * self.chromosome_length
                # 修复：确保binary_str是正确的二进制字符串
                binary_str = ''.join(str(int(bit)) for bit in population[i, start:end])
                decimal = int(binary_str, 2)
                
                # 映射到[-20, 20]区间
                min_val, max_val = self.bounds
                decoded[i, j] = min_val + (max_val - min_val) * decimal / (2**self.chromosome_length - 1)
        return decoded
    
    def calculate_fitness(self, x: np.ndarray) -> np.ndarray:
        """计算种群中每个个体的适应度"""
        fitness_values = np.array([fitness_function(individual) for individual in x])
        # 由于是最小化问题，需要将适应度取反
        return -fitness_values
    
    def select_parents(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        # 保留精英个体
        elite_indices = np.argsort(fitness)[-self.elite_size:]
        elite_population = population[elite_indices]
        
        # 对剩余位置进行锦标赛选择
        tournament_size = 3
        selected = np.zeros((self.pop_size - self.elite_size, self.total_length))
        for i in range(self.pop_size - self.elite_size):
            tournament_idx = np.random.choice(self.pop_size, tournament_size, replace=False)
            winner_idx = tournament_idx[np.argmax(fitness[tournament_idx])]
            selected[i] = population[winner_idx]
        
        return np.vstack((elite_population, selected))
    
    def crossover(self, parents: np.ndarray) -> np.ndarray:
        offspring = parents.copy()
        for i in range(self.elite_size, self.pop_size-1, 2):
            if np.random.random() < self.crossover_rate:
                # 对每个维度分别进行交叉
                for d in range(self.dim):
                    start = d * self.chromosome_length
                    end = (d + 1) * self.chromosome_length
                    # 两点交叉
                    points = sorted(np.random.choice(self.chromosome_length-1, 2, replace=False) + 1)
                    for j in range(points[0], points[1]):
                        offspring[i, start+j], offspring[i+1, start+j] = \
                            offspring[i+1, start+j], offspring[i, start+j]
        return offspring
    
    def mutate(self, population: np.ndarray) -> np.ndarray:
        # 保护精英个体不参与变异
        mutation_mask = np.random.random(population[self.elite_size:].shape) < self.mutation_rate
        population[self.elite_size:][mutation_mask] = 1 - population[self.elite_size:][mutation_mask]
        return population
    
    def check_convergence(self, fitness_history: List[float], window: int = 30) -> bool:
        if len(fitness_history) < window:
            return False
        recent_fitness = fitness_history[-window:]
        return np.std(recent_fitness) < self.convergence_threshold
    
    def run(self) -> Tuple[List[float], List[np.ndarray]]:
        start_time = time()
        population = self.initialize_population()
        best_fitness_history = []
        best_x_history = []
        
        for generation in range(self.generations):
            x = self.decode_chromosomes(population)
            fitness = self.calculate_fitness(x)
            
            best_idx = np.argmax(fitness)
            best_fitness_history.append(-fitness[best_idx])  # 转回原始函数值
            best_x_history.append(x[best_idx])
            
            if self.check_convergence(best_fitness_history):
                self.convergence_generation = generation
                break
                
            parents = self.select_parents(population, fitness)
            offspring = self.crossover(parents)
            population = self.mutate(offspring)
        
        self.execution_time = time() - start_time
        return best_fitness_history, best_x_history
    
    def get_statistics(self) -> dict:
        return {
            "收敛代数": self.convergence_generation,
            "运行时间": f"{self.execution_time:.4f}秒"
        }

def evaluate_algorithm(n_runs: int = 10) -> Tuple[np.ndarray, float, List[List[float]]]:
    best_solutions = []
    best_fitness_values = []
    convergence_gens = []
    execution_times = []
    all_history = []  # 记录所有运行的历史
    
    for i in range(n_runs):
        ga = GeneticAlgorithm()
        best_fitness_history, best_x_history = ga.run()
        stats = ga.get_statistics()
        
        best_idx = np.argmin(best_fitness_history)
        best_solutions.append(best_x_history[best_idx])
        best_fitness_values.append(best_fitness_history[best_idx])
        convergence_gens.append(stats["收敛代数"])
        execution_times.append(float(stats["运行时间"].replace("秒", "")))
        
        # 将历史记录填充到最大代数
        padded_history = best_fitness_history + [best_fitness_history[-1]] * (ga.generations - len(best_fitness_history))
        all_history.append(padded_history[:ga.generations])
    
    # 计算统计指标
    best_solution = best_solutions[np.argmin(best_fitness_values)]
    theoretical_minimum = 20.0
    absolute_error = np.abs(min(best_fitness_values) - theoretical_minimum)
    relative_error = absolute_error / theoretical_minimum * 100
    
    print("\n=== 算法性能统计（{}次运行） ===".format(n_runs))
    print(f"最优解范数平均值: {np.mean([np.linalg.norm(x) for x in best_solutions]):.6f}")
    print(f"最优解范数标准差: {np.std([np.linalg.norm(x) for x in best_solutions]):.6f}")
    print(f"最优适应度平均值: {np.mean(best_fitness_values):.6f} ± {np.std(best_fitness_values):.6f}")
    print(f"绝对误差: {absolute_error:.6f}")
    print(f"相对误差: {relative_error:.6f}%")
    print(f"平均收敛代数: {np.mean(convergence_gens):.1f} ± {np.std(convergence_gens):.1f}")
    print(f"平均运行时间: {np.mean(execution_times):.4f} ± {np.std(execution_times):.4f}秒")
    
    return best_solution, min(best_fitness_values), all_history

def plot_results(ga: GeneticAlgorithm, best_fitness_history: List[float], 
                all_history: List[List[float]], best_solution: np.ndarray) -> None:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 优化过程（对数刻度）
    ax1 = fig.add_subplot(221)
    ax1.semilogy(best_fitness_history, 'b-', label='当前运行')
    ax1.set_title('优化过程 (对数刻度)')
    ax1.set_xlabel('代数')
    ax1.set_ylabel('目标函数值 (log)')
    if ga.convergence_generation:
        ax1.axvline(x=ga.convergence_generation, color='r', linestyle='--',
                   label=f'收敛于第{ga.convergence_generation}代')
    ax1.grid(True)
    ax1.legend()
    
    # 2. 所有运行的收敛曲线
    ax2 = fig.add_subplot(222)
    all_history_array = np.array(all_history)
    mean_history = np.mean(all_history_array, axis=0)
    std_history = np.std(all_history_array, axis=0)
    generations = range(len(mean_history))
    ax2.plot(generations, mean_history, 'b-', label='平均收敛曲线')
    ax2.fill_between(generations, 
                     mean_history - std_history,
                     mean_history + std_history,
                     alpha=0.2, color='b', label='标准差范围')
    ax2.set_title('多次运行收敛曲线')
    ax2.set_xlabel('代数')
    ax2.set_ylabel('目标函数值')
    ax2.grid(True)
    ax2.legend()
    
    # 3. 最优解的各维度值分布
    ax3 = fig.add_subplot(223)
    dimensions = range(1, len(best_solution) + 1)
    ax3.bar(dimensions, best_solution, alpha=0.6)
    ax3.set_title('最优解各维度值分布')
    ax3.set_xlabel('维度')
    ax3.set_ylabel('取值')
    ax3.grid(True)
    
    # 4. 误差分析箱线图
    ax4 = fig.add_subplot(224)
    final_values = all_history_array[:, -1] - 20  # 减去理论最小值
    ax4.boxplot(final_values)
    ax4.set_title('最终误差分布')
    ax4.set_ylabel('与理论最小值的误差')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    # 多次运行统计
    best_solution, best_fitness, all_history = evaluate_algorithm(n_runs=10)
    
    # 单次运行展示
    ga = GeneticAlgorithm()
    best_fitness_history, best_x_history = ga.run()
    stats = ga.get_statistics()
    
    print("\n=== 最佳运行结果 ===")
    print(f"最优解: {best_solution}")
    print(f"最优解范数: {np.linalg.norm(best_solution):.6f}")
    print(f"最小值: {best_fitness:.6f}")
    print(f"收敛代数: {stats['收敛代数']}")
    print(f"运行时间: {stats['运行时间']}")
    
    # 计算理论最小值的相对误差
    relative_error = abs(best_fitness - 20.0) / 20.0 * 100
    print(f"相对误差: {relative_error:.6f}%")
    
    # 绘制结果
    plot_results(ga, best_fitness_history, all_history, best_solution)

if __name__ == "__main__":
    main() 