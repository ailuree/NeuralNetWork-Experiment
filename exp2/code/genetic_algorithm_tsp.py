import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# 随机生成城市坐标
def generate_cities(num_cities: int) -> np.ndarray:
    return np.random.rand(num_cities, 2) * 100  # 生成0到100之间的坐标

# 计算两城市之间的欧氏距离
def calculate_distance(city1: np.ndarray, city2: np.ndarray) -> float:
    return np.linalg.norm(city1 - city2)

# 计算路线的总距离
def total_distance(route: List[int], cities: np.ndarray) -> float:
    distance = 0.0
    for i in range(len(route)):
        distance += calculate_distance(cities[route[i]], cities[route[(i + 1) % len(route)]])
    return distance

# 遗传算法类
class GeneticAlgorithmTSP:
    def __init__(self, cities: np.ndarray, pop_size: int = 100, generations: int = 500, mutation_rate: float = 0.01):
        self.cities = cities
        self.num_cities = len(cities)
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()
    
    def initialize_population(self) -> List[List[int]]:
        return [np.random.permutation(self.num_cities).tolist() for _ in range(self.pop_size)]
    
    def calculate_fitness(self, population: List[List[int]]) -> List[float]:
        return [1 / total_distance(route, self.cities) for route in population]
    
    def select_parents(self, fitness: List[float]) -> List[List[int]]:
        fitness = np.array(fitness)
        probs = fitness / fitness.sum()
        indices = np.random.choice(self.pop_size, size=self.pop_size, p=probs)
        return [self.population[i] for i in indices]
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        start, end = sorted(np.random.choice(self.num_cities, 2, replace=False))
        child = [-1] * self.num_cities
        child[start:end] = parent1[start:end]
        
        pointer = end
        for city in parent2:
            if city not in child:
                if pointer >= self.num_cities:
                    pointer = 0
                child[pointer] = city
                pointer += 1
        return child
    
    def mutate(self, route: List[int]) -> List[int]:
        for i in range(self.num_cities):
            if np.random.random() < self.mutation_rate:
                j = np.random.randint(0, self.num_cities)
                route[i], route[j] = route[j], route[i]
        return route
    
    def run(self) -> Tuple[List[int], float]:
        best_route = None
        best_distance = float('inf')
        best_distances = []
        
        for generation in range(self.generations):
            fitness = self.calculate_fitness(self.population)
            self.population = self.select_parents(fitness)
            
            next_generation = []
            for i in range(0, self.pop_size, 2):
                parent1, parent2 = self.population[i], self.population[i+1]
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)
                next_generation.extend([self.mutate(child1), self.mutate(child2)])
            
            self.population = next_generation
            
            current_best_distance = 1 / max(fitness)
            best_distances.append(current_best_distance)
            if current_best_distance < best_distance:
                best_distance = current_best_distance
                best_route = self.population[np.argmax(fitness)]
            
            if generation % 50 == 0:
                print(f"Generation {generation}: Best Distance = {best_distance:.2f}")
        
        return best_route, best_distance, best_distances

# 主函数
def main():
    num_cities = 15
    cities = generate_cities(num_cities)
    ga_tsp = GeneticAlgorithmTSP(cities)
    best_route, best_distance, best_distances = ga_tsp.run()
    
    print("\n最佳路线:", best_route)
    print("最短距离:", best_distance)
    
    # 绘制结果
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 6))
    plt.scatter(cities[:, 0], cities[:, 1], c='red')
    for i, city in enumerate(cities):
        plt.text(city[0], city[1], str(i), fontsize=12, ha='right')
    
    route_cities = cities[best_route + [best_route[0]]]
    plt.plot(route_cities[:, 0], route_cities[:, 1], 'b-', marker='o')
    plt.quiver(route_cities[:-1, 0], route_cities[:-1, 1], 
               route_cities[1:, 0] - route_cities[:-1, 0], 
               route_cities[1:, 1] - route_cities[:-1, 1], 
               angles='xy', scale_units='xy', scale=1, color='blue', width=0.003)
    plt.scatter(route_cities[0, 0], route_cities[0, 1], c='green', s=100, label='起点')
    plt.scatter(route_cities[-1, 0], route_cities[-1, 1], c='orange', s=100, label='终点')
    plt.title('最佳旅行路线')
    plt.xlabel('X 坐标')
    plt.ylabel('Y 坐标')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 绘制误差分析
    plt.figure(figsize=(10, 5))
    plt.plot(best_distances, label='最佳距离')
    plt.title('距离随代数变化')
    plt.xlabel('代数')
    plt.ylabel('距离')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main() 