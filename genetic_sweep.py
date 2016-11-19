import numpy as np
import json
import random
# Since most of the work is done in a subprocess, just use threading to
# simplify sharing, which by the way is caring
import multiprocessing.dummy as multiprocessing
import os
from multiprocessing import cpu_count
from copy import deepcopy
from subprocess import check_output

MUTATION_PROB = 0.1
starting_params = {
    '--pa'   : 0.1,
    '--pp'   : 0.1,
    '--pw'   : 0.1,
    '--pn'   : 0.1,
    '--pi'   : 0.1,
    '--pe'   : 0.1,
    '--pt'   : 0.1,
    '--pa_ms': 0.5,
    '--pa_me': 1,
    '--pa_mp': np.log(0.25),
    '--pp_ms': 0.3,
    '--pp_me': 1,
    '--pp_mp': np.log(0.25),
    '--pt_ms': 0,
    '--pt_me': 1,
    '--pt_mp': np.log(1),
    '--pw_ms': 1,
    '--pw_me': 0,
    '--pw_mp': np.log(2),
    '--pi_ms': 0.5,
    '--pi_me': 1,
    '--pi_mp': np.log(1),
    '--pe_ms': 0,
    '--pe_me': 1,
    '--pe_mp': np.log(2),
    '--pn_ms': 1,
    '--pn_me': 0,
    '--pn_mp': np.log(2),
}

def get_bounds(key):
    bounds = [0, 1]
    if 'mp' in key:
        bounds = [-2, 2]
    return bounds

def get_final_territory(fn):
    with open(fn, 'r') as f:
        j = json.load(f)
    arr = np.asarray(j['frames'][-1])[:, :, 0]
    return [np.sum(arr==i) / arr.size for i in range(j['num_players'] + 1)]

def cross(p1, p2):
    pnew = {}
    for key in p1:
        r = random.randint(1, 4)
        if r == 1: pnew[key] = p1[key]
        elif r == 2: pnew[key] = p2[key]
        elif r >= 3: pnew[key] = (p1[key] + p2[key]) / 2
    return pnew

def mutate(p, prob=MUTATION_PROB):
    for key in p:
        if random.random() < prob:
            p[key] = random.uniform(*get_bounds(key))
    return p

def run_str(p):
    pp = deepcopy(p)
    for k in pp:
        if 'mp' in k: pp[k] = np.exp(pp[k])
    return '"python3 ../baselines/potential_bot.py {}"'.format(
        " ".join("{} {}".format(*v) for v in p.items()))

def play_game(width, height, players):
    np = len(players)
    cmd = '../halite -t -q -d "{} {}" {}'.format(
        width, height, " ".join(run_str(p) for p in players))
    output = check_output(cmd, shell=True).decode().split('\n')
    rest = ''.join(output[np * 2 + 1:]).strip()
    errored = None
    if rest != '':
        print("ERROR: " + rest)
        errored = [int(x) for x in rest.split(' ') if 'log' not in x]
    results = output[np + 1 : np * 2 + 1]
    territory = get_final_territory(output[np].split(' ')[0])
    ranking = [int(x[-1]) for x in results]
    scores = [1/(ranking[i]+1-territory[i+1]) for i in range(np)]
    if errored is not None:
        for i in errored:
            scores[i-1] = 0
    return scores


def worker(q):
    while True:
        args = q.get()
        accumulate(*args)
        q.task_done()

l = multiprocessing.Lock()
def accumulate(width, height, players, population, scores, ngames):
    results = play_game(width, height, [population[i] for i in players])
    with l:
        for e, i in enumerate(players):
            scores[i] += results[e]
        for p in players:
            ngames[p] += 1
        print(min(ngames), np.asarray(scores) / np.asarray(ngames))

def simulate(population, queue=None):
    print("Iteration...")
    i = 0
    scores = [0 for i in range(len(population))]
    ngames = [0 for i in range(len(population))]
    while True:
        if min(ngames) > 2:
            break
        i += 1
        nplayers = random.randint(2, 6)
        width = random.randint(20, 50)
        height = random.randint(20, 50)

        players = np.random.choice(range(len(population)), size=nplayers, replace=False)

        if queue is not None:
            queue.put((width, height, players, population, scores, ngames))
        else:
            accumulate(width, height, players, population, scores, ngames)

    print("Wating for queue of size {} to finish..".format(queue.qsize()))
    if queue is not None:
        queue.join()
    return np.asarray(scores) / np.asarray(ngames)

if __name__ == "__main__":
    np.set_printoptions(precision=3)
    os.chdir("sweep")
    if os.path.exists('../sweep.out'):
        with open("../sweep.out", 'r') as f:
            data = list(reversed(f.readlines()))
        v = ""
        for i in data:
            if "ITERATION" in i:
                break
            v = i
        population = eval(v)
        children = []
        for i in range(15):
            p1, p2 = np.random.choice(population, replace=False, size=2)
            children.append(mutate(cross(p1, p2)))
        population += children
        print("Previous sweep loaded!")
    else:
        population = [starting_params] + \
                [mutate(deepcopy(starting_params), 0.2) for i in range(19)]
        print("Starting from scratch!")

    # The queue can hold 3 additional objects
    queue = multiprocessing.JoinableQueue(3)
    for i in range(min(cpu_count(), min(cpu_count(), 20))):
        multiprocessing.Process(target=worker, args=(queue,)).start()
    while True:
        scores = simulate(population, queue)
        bad = np.argpartition(scores, -5)[:-5]
        scores[bad] = 0
        print(scores)
        children = []
        for i in range(15):
            p1, p2 = np.random.choice(population, p=scores/scores.sum(), replace=False, size=2)
            children.append(mutate(cross(p1, p2)))
        print([y for (y, x) in sorted(zip(scores, population), key=lambda x: x[0], reverse=True)][:5])
        population = [x for (y, x) in sorted(zip(scores, population), key=lambda x: x[0], reverse=True)][:5] + children
        with open("../sweep.out", 'a') as f:
            f.write("== ITERATION ==\n")
            f.write(str(population[:5]) + "\n")
            f.write("\n".join(run_str(x) for x in population[:5]) + "\n")
            f.write("\n\n")
