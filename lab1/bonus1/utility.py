import tqdm


def monte_carlo_success(policy, env, N=1000):
    method = "ValIter"
    n_win = 0
    for i in tqdm.tqdm(range(N)):
        path = env.simulate((0, 0), policy, method)
        if env.win(env.map[path[-1]]):
            n_win += 1
    print(n_win/N)
