# Paper link: https://arxiv.org/abs/2102.10200

# imports
import os
import numpy as np
import matplotlib.pyplot as plt

# function to calculate kl divergence between two bernoulli distribution
def kl_div(p, q):
    if p == 0:
        return -np.log(1-q)
    if p == 1:
        return -np.log(q)
    return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

# function to calculate the max q for agent m and arm k using binary search
def find_q(m, k, rhs, eps, mu_hat, N):    
    emp_mean = mu_hat[m][k]
    l = emp_mean
    r = 1

    if(N[m][k] == 0):
        return (l+r)/2
    kl_div_bound = rhs/N[m][k]

    while(r-l > eps):
        mid = (l+r)/2
        div = kl_div(emp_mean, mid)
        if(div < kl_div_bound):
            l = mid
        else:
            r = mid
    return (l+r)/2

# run simulation
def run_klucb(num_of_players, num_of_arms, time_steps, true_mean_reward_min, true_mean_reward_max, randomized):
    # initialize variables
    M = num_of_players
    K = num_of_arms
    mu = np.linspace(true_mean_reward_max, true_mean_reward_min, num=K) # equally spaced true mean rewards
    T = time_steps
    N = np.zeros((M, K))            # N[m][k] = number of times agent m pulled arm k
    mu_hat = np.zeros((M, K))       # mu_hat[m][k] = running averge of rewards for agent m and arm k
    regret = np.zeros(T+1)          # cumulative regret per time step
    eps = 1e-3                      # precision used while finding q
    c = 3                           # fn(t) = ln(t) + c*ln(ln(t))

    # to show progress on terminal
    print("Simulating for:")
    print(f"M={M}, K={K}, T={T}, randomized={randomized}")
    print("mu=")
    print(mu)
    print()

    # run for T time steps
    for t in range(1, T+1):
        arm_pulled = np.zeros(M)    # arm_pulled[i] = arm pulled by agent i
        arm_count = np.zeros(K)     # arm_count[i] = number of agents that pull arm i
        curr_regret = 0             # additional regret for this time step
        
        # fn = fn(t), not defined for t = 1
        if t == 1:
            fn = -10
        else:
            fn = np.log(t) + c*np.log(np.log(t))

        # find which arm is pulled by each agent
        for m in range(M):
            q_vec = np.zeros(K)                     # q[i] = ucb index for (m, k)
            for k in range(K):        
                q_vec[k] = find_q(m, k, fn, eps, mu_hat, N)
                if(randomized):                     # add randomization to ucb index
                    Z = np.random.normal()
                    q_vec[k] += Z/t
            arm_pulled[m] = np.argmax(q_vec)        # chosen arm has highest ucb index
            arm_count[(int)(arm_pulled[m])] += 1    # update chosen arm count

        # update values
        for m in range(M):
            indicator = 0                           # collision indicator
            if(arm_count[(int)(arm_pulled[m])] > 1):
                indicator = 1
            reward = np.random.binomial(1, mu[(int)(arm_pulled[m])])*(1-indicator)  # observed reward
            
            # value updates
            mu_hat[m][(int)(arm_pulled[m])] = (N[m][(int)(arm_pulled[m])] * mu_hat[m][(int)(arm_pulled[m])] + reward) / (N[m][(int)(arm_pulled[m])] + 1)
            N[m][(int)(arm_pulled[m])] += 1
            curr_regret += (mu[m] - mu[(int)(arm_pulled[m])]*(1-indicator))

        # update cumulative regret
        regret[t] = regret[t-1] + curr_regret
    return regret

# function to get a single plot
def plot_simulation(num_of_players, num_of_arms, time_steps, true_mean_reward_min, true_mean_reward_max):
    klucb = run_klucb(num_of_players, num_of_arms, time_steps, true_mean_reward_min, true_mean_reward_max, False)
    rnd_klucb = run_klucb(num_of_players, num_of_arms, time_steps, true_mean_reward_min, true_mean_reward_max, True)
    plt.plot(np.linspace(0, time_steps, time_steps+1), klucb, label="Selfish KL-UCB")
    plt.plot(np.linspace(0, time_steps, time_steps+1), rnd_klucb, label="Rnd-Selfish KL-UCB")
    plt.xlabel("t")
    plt.ylabel("Cumulative Regret")
    plt.title(f"M={num_of_players}, K={num_of_arms}, mu_min={true_mean_reward_min}, mu_max={true_mean_reward_max}")
    plt.legend(loc=(1.04, 0.5))
    plt.savefig(f"./results/M_{num_of_players}_K_{num_of_arms}_mu_min_{true_mean_reward_min}_mu_max_{true_mean_reward_max}.png", bbox_inches='tight')
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    plt.clf()

# function to get multiple plots
def plot_multiple_simulations(agent_arm_tuple_list, true_reward_tuple_list, time_steps):
    for agent_arm_tuple in agent_arm_tuple_list:
        for true_reward_tuple in true_reward_tuple_list:
            plot_simulation(agent_arm_tuple[0], agent_arm_tuple[1], time_steps, true_reward_tuple[0], true_reward_tuple[1])

# function to get regret histogram
def plot_regret_histogram(num_of_runs, num_of_players, num_of_arms, time_steps, true_mean_reward_min, true_mean_reward_max):
    klucb = []
    rnd_klucb = []
    for _ in range(num_of_runs):
        klucb.append(run_klucb(num_of_players, num_of_arms, time_steps, true_mean_reward_min, true_mean_reward_max, False)[-1])
        rnd_klucb.append(run_klucb(num_of_players, num_of_arms, time_steps, true_mean_reward_min, true_mean_reward_max, True)[-1])
    bins = np.linspace(0, 1100, 50)
    plt.hist(klucb, bins, rwidth=0.7, alpha=0.8, label="Selfish KL-UCB")
    plt.hist(rnd_klucb, bins, rwidth=0.7, alpha=0.8, label="Rnd-Selfish KL-UCB")
    plt.xlabel("Total Cumulative Regret")
    plt.ylabel("Number of Runs")
    plt.title(f"Runs={num_of_runs}, M={num_of_players}, K={num_of_arms}, mu_min={true_mean_reward_min}, mu_max={true_mean_reward_max}, T={time_steps}")
    plt.legend(loc=(1.04, 0.5))
    plt.savefig(f"./results/hist_runs_{num_of_runs}_M_{num_of_players}_K_{num_of_arms}_mu_min_{true_mean_reward_min}_mu_max_{true_mean_reward_max}.png", bbox_inches='tight')
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    plt.clf()

# create directory to save results
if not os.path.exists("./results"):
    os.makedirs("./results")

# run simulations
plot_multiple_simulations([(2, 3), (2, 5), (5, 10), (10, 15)], [(0.1, 0.2), (0.8, 0.9), (0.01, 0.99)], 10000)
plot_regret_histogram(500, 2, 2, 1000, 0.1, 0.9)
