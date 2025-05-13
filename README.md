# COIT29224- Evolutionary Computational:<br>
The current project is a realization of (μ/μ, λ)-Evolution Strategy (ES) to solve the Rastrigin function as a difficult benchmark task, which has a lot of local minima. <br>

Rastrigin function is defined in the 10 dimensions continuous space with a global minimum on the origin. Usual gradient-based approaches do not succeed on such problems, because these get stuck in local minima. As opposed to this, Evolution Strategies employ population-level stochastic methods that better help to explore a complex multimodal landscape.  <br>

The (μ/μ, λ)-ES algorithm works by initializing randomly generated parent population. λ off spring are produced from recombination and Gaussian mutation in each generation. The μ best offsprings are selected on the basis of fitness to make the succeeding generation. Such an implementationemploys intermediate recombination as well as fixed mutation parameters, thus providing constant convergence without losing exploration.  <br>

That method was selected for its robustness and ease of applying continuous- optimisation problems. Different from the Genetic Algorithms (GAs), ES deals with the real-valued vectors and attaches more weight on self-adaptive mutation than the discrete crossover, with more stable convergence in the continuous domains. It also bypasses the weaknesses of the gradient-based approaches since it is not dependent on any derivative information. <br>

The code gives convergence plot for fitness improvements during generations. It illustrates the effective manner in which the ES can use to reach close to global optima even with the deceptive nature of the Rastrigin’s landscape. This makes it a good candidate for real-world applications with noise, non-convex and high dimensions optimizing problems.
