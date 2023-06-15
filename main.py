import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from alive_progress import alive_bar
import csv

matplotlib.use('TkAgg')


def read_csv(file_name, skip_header=True):
    # Reads csv files that are formatted as feq, z.real, z.imag
    with open(file_name, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        if skip_header:
            next(reader)  # skip header
        data = np.array([[float(row[0]), float(row[1]), float(row[2])] for row in reader])
        freq, real, imag = data[:, 0], data[:, 1], data[:, 2]
        z = real + 1j * imag
        return freq, z


class DifferentialEvolution:
    # Differential Evolution algorithm
    def __init__(self, bounds, exp_data, pop_size=100, max_iter=100, fit='warburg', mutf=0.7, crossr=0.8):
        self.bounds = bounds
        self.exp_data = exp_data
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.fit = fit
        self.mutf = mutf
        self.crossr = crossr
        self.dim = len(bounds)

        self.populations = np.zeros((max_iter, pop_size, self.dim))
        self.fitnesses = np.zeros((max_iter, pop_size))
        self.solution = (None, None)  # (best fitness, best individual)

    def solve(self):
        # Main differential evolution loop

        # Initialize population
        self.populations[0] = np.random.uniform([i[0] for i in self.bounds.values()],
                                                [i[1] for i in self.bounds.values()], (self.pop_size, self.dim))

        with alive_bar(self.max_iter) as bar:
            for gen in range(self.max_iter):
                # Evaluate fitness
                self.fitnesses[gen] = np.asarray([self.objective_function(vec) for vec in self.populations[gen]])

                fittest_idx = np.argmin(self.fitnesses[gen])
                fittest_vec = self.populations[gen][fittest_idx]

                # Iterate over population
                for i in range(self.pop_size):
                    # Select two random vectors distinct from current and fittest
                    indices = [idx for idx in range(self.pop_size) if idx != i and idx != fittest_idx]
                    i1, i2 = np.random.choice(indices, 2, replace=False)

                    # Mutate
                    mutant = fittest_vec + self.mutf * (self.populations[gen][i1] - self.populations[gen][i2])

                    # Crossover
                    cross_points = np.random.rand(self.dim) <= self.crossr
                    if not np.any(cross_points):  # Ensure at least one parameter is changed
                        cross_points[np.random.randint(0, self.dim)] = True

                    trial = np.where(cross_points, mutant, self.populations[gen][i])

                    # Penalize trial vector if it is out of bounds
                    for j, key in enumerate(self.bounds.keys()):
                        lower_bound, upper_bound = self.bounds[key]
                        if trial[j] < lower_bound or trial[j] > upper_bound:
                            trial_fitness = 1e9999
                            break
                        else:  # Executes if no break occurs
                            trial_fitness = self.objective_function(trial)

                    # Select
                    if gen + 1 < self.max_iter:  # Make sure we don't go out of bounds
                        if trial_fitness < self.fitnesses[gen][i]:  # Check if trial is better
                            self.populations[gen + 1][i] = trial
                            self.fitnesses[gen + 1][i] = trial_fitness
                        else:  # Keep current vector
                            self.populations[gen + 1][i] = self.populations[gen][i]
                            self.fitnesses[gen + 1][i] = self.fitnesses[gen][i]
                bar()

        # Find best solution
        fittest_idx = np.argmin(self.fitnesses[-1])
        result = self.populations[-1][fittest_idx]
        result_fitness = self.fitnesses[-1][fittest_idx]
        assert result_fitness == np.min(self.fitnesses[-1])
        self.solution = (result, result_fitness)

    def objective_function(self, vec):
        # Returns the fitness of a vector
        if vec is None:
            return 100
        freqs = self.exp_data[0]
        w = 2 * np.pi * freqs
        z_experimental = self.exp_data[1]
        z_calculated = self.z_from_w(vec, w)
        delta = np.linalg.norm((z_experimental - z_calculated))  # Scale by magnitude
        return np.sqrt(np.mean(delta ** 2))

    def z_from_w(self, vec, w):
        if self.fit == 'warburg':
            return self.warburg(vec, w)
        elif self.fit == 'havriliak-negami':
            return self.havriliak_negami(vec, w)
        elif self.fit == 'cpe_parallel':
            return self.cpe_parallel(vec, w)
        elif self.fit == 'h_n_cpe_parallel':
            return self.h_n_cpe_parallel(vec, w)
        else:
            raise ValueError("Invalid fit type")

    @staticmethod
    def havriliak_negami(vec, w):
        # Returns the impedance of a Havriliak-Negami circuit
        rhf, rlf, tau, alpha, nu, beta, q = vec
        z = rhf + (rlf - rhf) / ((1 + (1j * w * tau) ** nu) ** beta) + (1 / (q * (1j * w) ** alpha))
        return z

    @staticmethod
    def cpe_parallel(vec, w):
        q1, alpha1, r = vec
        # Two CPEs in parallel with a resistor in series
        """  z1  
        --------<<-------
        |               |
        ---------/\/\----
            z2     R
        """
        z = 1 / ((q1 * (1j * w) ** alpha1) + 1 / r)
        return z

    @staticmethod
    def h_n_cpe_parallel(vec, w):
        rhf, rlf, tau, alpha, nu, beta, q, q1, alpha1, r = vec
        z = rhf + (rlf - rhf) / ((1 + (1j * w * tau) ** nu) ** beta) + (1 / (q * (1j * w) ** alpha)) + (1 / ((q1 * (1j * w) ** alpha1) + 1 / r))
        return z

    @staticmethod
    def warburg(vec, w):
        r1, r2, c, alpha, s1 = vec
        z = r1 + 1 / (((1j * w) ** alpha) * c + 1 / (r2 + s1 / np.sqrt(1j * w / 2)))
        return z

    def plot_solution(self, vec, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        freqs = self.exp_data[0]
        zreal_exp, zimag_exp = self.exp_data[1].real, self.exp_data[1].imag
        ax.scatter(zreal_exp, -zimag_exp, label='Experimental Data', color='g', marker='x')
        for i, z in enumerate(self.exp_data[1]):
            if i % 10 == 0:
                ax.annotate(f"{freqs[i]} Hz", (z.real - 3.3, -z.imag + .3))

        # Plot calculated solution
        w = 2 * np.pi * freqs
        z = self.z_from_w(vec, w)
        ax.title.set_text("Nyquist plot}")
        ax.scatter(np.real(z), -np.imag(z), label='Calculated Solution', color='r', marker='v')
        ax.set_xlabel("Re(Z)")
        ax.set_ylabel("-Im(Z)")
        ax.grid()
        ax.legend()
        if ax is not None:
            plt.show()
        return fig, ax

    def plot_fitness(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        # else:
        #     fig = ax.figure
        averages = [np.average(gen) for gen in self.fitnesses]

        ax.title.set_text("Evolution of average population Fitness")
        ax.semilogy(averages)
        ax.set_xlabel("Generation")
        ax.set_ylabel("RMSE")
        ax.grid()
        ax.set_xlim((0, self.max_iter - 1))

        if ax is not None:
            plt.show()
        return fig, ax

    def display_plot_result(self):
        # Displays the values
        print(f"Fitting {self.dim} parameters to {len(self.exp_data[0])} data points")
        print(f"RMSE: {self.solution[1]}")
        for idx, key in enumerate(self.bounds.keys()):
            print(f"{key}: {self.solution[0][idx]}")
        # plots the solution and fitness
        self.plot_solution(self.solution[0])


if __name__ == "__main__":
    # Warburg Test
    # data = read_csv('input/sheet.csv')
    #
    # b = {'R1': (0, 1000),
    #      'R2': (0, 1000),
    #      'C': (0, 1e-3),
    #      'alpha': (0, 1),
    #      's1': (0, 1000)}
    #
    # DE = DifferentialEvolution(b, data, fit='warburg', max_iter=200)
    # DE.solve()
    # DE.display_plot_result()
    # DE.plot_fitness()

    # Havriliak-Negami Test
    # data = read_csv('input/CPE.csv')
    # b = {'Rhf': (0, 10),
    #      'Rlf': (0, 10),
    #      'tau': (0, 1),
    #      'alpha': (0, 1),
    #      'nu': (0, 1),
    #      'beta': (0, 1),
    #      'q': (0, 100e-4)}
    #
    # DE = DifferentialEvolution(b, data, fit='havriliak-negami', max_iter=200, pop_size=100)
    # DE.solve()
    # DE.plot_fitness()
    # DE.display_plot_result()

    # CapaPure Test
    # data = read_csv('input/CapaPure.csv')
    # b = {'Rhf': (0, 1),
    #      'Rlf': (0, 1),
    #      'tau': (0, 500e-6),
    #      'alpha': (1, 1),
    #      'beta': (0, 1),
    #      'q': (0, 10e-6)}
    #
    # DE = DifferentialEvolution(b, data, fit='havriliak-negami', max_iter=200, pop_size=200)
    # DE.solve()
    # DE.plot_fitness()
    # DE.display_plot_result()

    # CPE Parallel Test
    data = read_csv('input/ZLowFreq_salvador_10mM.csv')
    b = {'q1': (0, 1e-3),  # CPE Part
         'alpha1': (0, 1),
         'r': (0, 1e6)}

    DE = DifferentialEvolution(b, data, fit='cpe_parallel', max_iter=300, pop_size=200)
    DE.solve()
    DE.plot_fitness()
    DE.display_plot_result()

    # Full test: Havriliak-Negami + CPE Parallel - Check if conform wqith Rhf 98.8141 and Rlf 163.2486
    # data = read_csv('input/Z_salvador_full_10mM.csv')
    # b = {'Rhf': (0, 500),  # Havriliak-Negami Part
    #      'Rlf': (0, 500),
    #      'tau': (0, 500e-6),
    #      'alpha': (0, 1),
    #      'nu': (0, 1),
    #      'beta': (0, 1),
    #      'q': (0, 10e-3),
    #      'q1': (0, 1e-3),  # CPE Part
    #      'alpha1': (0, 1),
    #      'r': (0, 1e6)}
    #
    # DE = DifferentialEvolution(b, data, fit='h_n_cpe_parallel', max_iter=300, pop_size=200)
    # DE.solve()
    # DE.plot_fitness()
    # DE.display_plot_result()
