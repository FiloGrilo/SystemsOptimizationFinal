import pandas as pd
import numpy as np
import random
import math
import logging
import os
import time

# set file path for test cases
parent_dir = "/Users/joaomena/Documents/"
testcases_path = os.path.join(parent_dir, "testcases_seperation_tested")
test_file = "taskset__1643188013-a_0.1-b_0.1-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__0__tsk.csv"

# set path to log file
results_path = os.path.join(parent_dir, f"{test_file}_results")

if not os.path.exists(results_path):
    os.mkdir(results_path)

log_file = "project_results.log"
num = 1
if os.path.exists(os.path.join(results_path, log_file)):
    log_file = f"project_results{num}.log"
    while os.path.exists(os.path.join(results_path, f"project_results{num}.log")):
        num += 1
        log_file = f"project_results{num}.log"

# setup event logger
results_file = os.path.join(results_path, log_file)
logging.basicConfig(filename=results_file, filemode='w', format='%(asctime)s %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)

# ignore numpy warning
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

# set default values for SA temperature and cooling
def_temp = 20
def_cooling = 0.5

# set default polling server parameters
def_budget = 20
def_period = 25


class Task:
    """
    class used to represent tasks imported from the test cases
    """

    def __init__(self, task_dict):
        self.name = task_dict['name']
        self.duration = task_dict['duration']
        self.period = task_dict['period']
        self.type = task_dict['type']
        self.priority = task_dict['priority']
        self.deadline = task_dict['deadline']
        self.seperation = task_dict['seperation']


class SimAnnealingParams:
    """
    class used to define parameters used in the SA algorithm
    """

    def __init__(self, temperature, solution, cost, best_schedule, cooling_factor, norm_max):
        self.curr_temp = temperature
        self.cool = cooling_factor
        self.best_solution = solution
        self.best_cost = cost
        self.best_schedule = best_schedule
        self.iter = 1
        self.norm_max = norm_max


def tasks_parser(path, file):
    """
    get all tasks from the .csv file in the testcases folder and return them a list of Task objects
    :param file: specific test case file in path
    :param path: path to test cases file
    :return: list of Task objects with all tasks found
    """
    df = pd.read_csv(
        os.path.join(path, file),
        sep=';').to_dict(orient="index")  # read single .csv file and separate columns by ';'
    task_list = []
    for task in df:
        task_list.append(Task(df[task]))

    return task_list


def divisible_random(a, b, n):
    """
    generates a random number divisible by n in the interval between a and b
    :param a: lower limit of the interval
    :param b: higher limit of the interval
    :param n: number by which the new number should be divisible by
    :return: result of the operation
    """
    if b - a < n:
        raise Exception('{} is too big'.format(n))
    result = random.randint(a, b)
    while result % n != 0:
        result = random.randint(a, b)
    return result


def pdc(t_list, tt_hyperperiod, u, largest_deadline, deadline_list):
    """
    use processor demand criterion to determine if a set of tasks is schedulable or not
    :param t_list: array of tasks to consider
    :param tt_hyperperiod: hyperperiod from the tasks considered
    :param u: processor usage
    :param largest_deadline: largest deadline to consider
    :param deadline_list: list of deadlines from tasks considered
    :return: 0 if schedulable, 1 if not schedulable
    """

    # if processor usage is over 1, task set is not schedulable
    if u > 1:
        return 1

    # compute denominator for L*
    l_sum = 0
    for task in t_list:
        l_sum += (task.period - task.deadline) * (task.duration / task.period)
    # compute L*
    limit = l_sum / (1 - u)

    # condition value to determine time check points
    check_condition = min(tt_hyperperiod, max(largest_deadline, limit))

    # determine list of time points that must be checked using check_condition
    check_points = []
    for deadline in deadline_list:
        if deadline <= check_condition:
            check_points.append(deadline)

    # use demand bound function for every time check point to determine if task set is schedulable or not
    for t in check_points:
        dbf = 0
        for task in t_list:
            dbf += ((t + task.period - task.deadline) / task.period) * task.duration # demand bound function
        if dbf > t:
            return 1

    return 0


def edf_sim(t_list, ps_array):
    """
    schedule time triggered tasks and compute worst case response time
    :param t_list: array with time all tasks
    :param ps_array: array with polling servers
    :return: arrays with schedule made and worst case response time and hyperperiod
             or empty arrays if schedule is unfeasible within the deadline
    """
    # define arrays for computational time (C), deadline (D) and period (P)
    C = []
    D = []
    p = []
    U = 0

    # add tt tasks to array and fill arrays with parameters from all tasks
    max_d = 0  # save largest deadline
    tt_list = []
    for task in t_list:
        if task.type == "TT":
            tt_list.append(task)
            C.append(task.duration)
            D.append(task.deadline)
            if task.deadline > max_d:
                max_d = task.deadline
            p.append(task.period)
            U = U + task.duration / task.period

    # add polling servers to tt tasks array
    for ps in ps_array:
        tt_list.append(ps)
        C.append(ps.duration)
        D.append(ps.deadline)
        if ps.deadline > max_d:
            max_d = ps.deadline
        p.append(ps.period)
        U = U + ps.duration / ps.period

    # compute hyperperiod
    T = np.lcm.reduce(p)

    # initialize arrays of set length for r and wcrt
    r = np.zeros(len(tt_list))
    wcrt = np.zeros(len(tt_list))
    wcrt_changed = np.zeros(len(tt_list))

    # check if TT tasks are schedulable for EDF using processor demand criterion
    tt_valid = pdc(tt_list, T, U, max_d, D)
    if tt_valid > 0:
        print("Task set not schedulable")
        return [], [], T
    print("Task set schedulable")

    sigma = []
    t = 0

    # We go through each slot in the schedule table until T
    while t < T:
        state = 0
        i = 0
        for task in tt_list:
            if task.duration > 0 and task.deadline <= t:
                print('Deadline miss!')
                return [], [], T

            if t % task.period == 0:
                wcrt_changed[i] = 0
                r[i] = t
                task.duration = C[i]
                task.deadline = t + D[i]

            i += 1

        for task in tt_list:
            if task.duration != 0:  # there are still tasks that have computation time left
                state = 1
                break

        if state == 1:
            edf_name = edf(tt_list)
            sigma.append(edf_name)
            i = 0
            for task in tt_list:
                if edf_name == task.name:
                    task.duration -= 1

                if task.duration == 0 and task.deadline >= t and wcrt_changed[i] == 0 and edf_name == task.name:
                    if (t - r[i]) >= wcrt[i]:  # Check if the current WCRT is larger than the current maximum.
                        wcrt[i] = t - r[i]
                        wcrt_changed[i] = 1
                i += 1

        elif state == 0:
            sigma.append("idle")

        t += 1

    i = 0
    for task in tt_list:
        if C[i] > wcrt[i]:
            wcrt[i] = C[i]

        i += 1

        if task.duration > 0:
            print("Schedule is infeasible")
            return [], [], T

    return sigma, wcrt, T


def edf(tt_tasks):
    """
    get the name of the task with the earliest absolute deadline
    :param tt_tasks: array of tasks to consider
    :return: name of the task with the earliest absolute deadline
    """
    trade = 99999999999  # very high number
    name = ''
    for task in tt_tasks:
        if trade > task.deadline and task.duration != 0:
            trade = task.deadline
            name = task.name
    return name


def et_tasks_seperation(task_list, no_poll_srv):
    """
    Groups ET tasks by their separation value so that they can be separately scheduled by the polling server
    they are assigned to
    :param task_list: array of tasks to consider
    :param no_poll_srv: number of polling servers
    :return: list of the lists of et tasks assigned to each polling server
    """
    et_mask = []
    for task in task_list:
        et_mask.append((task.type == 'ET'))

    et_tasks = [task for task, y in zip(task_list, et_mask) if y]

    et_tasks_all_groups = []
    for i in range(no_poll_srv):
        et_task_group = []
        for task in et_tasks:
            if task.seperation == i + 1:
                et_task_group.append(task)
        et_tasks_all_groups.append(et_task_group)

    total_duration_groups = np.zeros((len(et_tasks_all_groups)))

    for index, task_groups in enumerate(et_tasks_all_groups):
        for task in task_groups:
            total_duration_groups[index - 1] += task.duration
    et_mask_seperation0 = []
    for task in et_tasks:
        et_mask_seperation0.append((task.seperation == 0))

    et_tasks_seperation0 = [task for task, y in zip(et_tasks, et_mask_seperation0) if y]

    for task in et_tasks_seperation0:
        index = np.argmin(total_duration_groups)
        et_tasks_all_groups[index].append(task)
        total_duration_groups[index] += task.duration

    return et_tasks_all_groups


def supply_poll_server(tt_schedule, poll_server):
    """
    Creates the schedule table of the polling server passed as argument
    :param tt_schedule: schedule of tt tasks
    :param poll_server: polling server
    :return: updated tt task schedule
    """
    supply_mask = np.empty((len(tt_schedule) + 1))
    name = "tPS{0}".format(poll_server + 1)
    for i in range(len(tt_schedule)):
        supply_mask[i] = (name == tt_schedule[i])
    return supply_mask


def et_schedule(et_tasks, supply_mask):
    """
    determine if event triggered tasks are schedulable and compute worst case response time
    :param supply_mask:
    :param et_tasks: list of tasks
    :return: bool for schedulability and list with worst case response times
    """

    period_list = []
    for task in et_tasks:
        period_list.append(task.period)
    hyperperiod = np.lcm.reduce(period_list)  # compute hyperperiod

    response_time = []
    for index, actual_task in zip(range(len(et_tasks)), et_tasks):
        t = 0  # current time
        supply = 0
        response_time.append(actual_task.deadline + 1)

        # Initialize the response time of τi to a value exceeding the deadline
        # because if it's not schedulable, it is already done to return False
        demand = 0
        for task in et_tasks:
            if task.priority >= actual_task.priority:
                demand += task.duration

        while t <= hyperperiod:
            # linear supply bound function that will be important to compute the WCRT
            # supply = alfa * (t - delta)
            # if supply < 0:
            #    supply = 0
            supply += supply_mask[t]
            # get subset of ET tasks that have higher priority than the current ET task
            if supply >= demand and supply != 0:  # if supply >=demand , we found the response time
                response_time[index] = t
                break
            t += 1
        if response_time[index] > actual_task.deadline:
            return False, response_time

    return True, response_time


def cost_function(tt_wcrt, et_wcrt_groups, et_sched):
    """
    compute cost function to determine quality of a solution
    :param et_wcrt_groups: event triggered tasks worst case response time
    :param tt_wcrt: time triggered tasks worst case response time
    :param et_sched: event triggered tasks schedulability
    :return: int value of computed cost
    """

    if len(tt_wcrt) == 0 or len(et_wcrt_groups) == 0:
        return 999999999999

    coefficient = 100
    sum_tt = 0
    for i in tt_wcrt:
        sum_tt += i
    sum_et = 0
    for et_wcrt in et_wcrt_groups:
        for i in et_wcrt:
            sum_et += i
    bool_var = et_sched
    # if the schedule for ET tasks is not possible, it will greatly impact in the cost
    cost = sum_tt / len(tt_wcrt) + sum_et * (1 + 2 * bool_var * coefficient) / len(et_wcrt_groups)

    return cost


def create_poll_srv(budgets, periods):
    """
    create polling server(s) with given parameters
    :param budgets: budget for the polling server
    :param periods: period for the polling server
    :return: list of polling servers created
    """
    ps_matrix = []
    i = 0
    for budget, period in zip(budgets, periods):
        ps_def_aux = {'name': "tPS{0}".format(i + 1), 'duration': budget, 'period': period, 'type': "TT", 'priority': 7,
                      'deadline': period, 'seperation': 0}
        ps_def = Task(ps_def_aux)
        ps_matrix.append(ps_def)
        i += 1
    return ps_matrix


def simulated_annealing(tt_tasks_wcrt, et_tasks_wcrt, tt_schedule, et_tasks_sched, candidate_solution, parameters,
                        hyperperiod, tt_schedule_bool):
    """
    compares cost of proposed solution to the best solution and returns random values to test again
    :param tt_tasks_wcrt: time triggered tasks worst case response time
    :param et_tasks_wcrt: event triggered tasks worst case response time
    :param tt_schedule: returned schedule from edf function
    :param et_tasks_sched: number of polling servers that are not shedulable for et tasks
    :param candidate_solution: list of parameters that make up the solution
    :param parameters: SimulatedAnnealingParams object
    :param hyperperiod: hyperperiod of all time triggered tasks including polling server
    :param tt_schedule_bool: bool that indicates if tt tasks are schedulable or not
    :return: new number of polling servers, budget and period randomly generated
    """

    # define limits for generated variables
    max_budget_variation = 40
    max_period_variation = 50

    if tt_schedule_bool == 1:

        # calculate cost with the parameters given
        candidate_cost = cost_function(tt_tasks_wcrt, et_tasks_wcrt, et_tasks_sched)

        print(f"Candidate cost: {int(candidate_cost)}")

        if candidate_cost < parameters.best_cost:  # update the best solution for lower cost

            parameters.best_cost = candidate_cost
            parameters.best_solution = candidate_solution
            parameters.best_schedule = tt_schedule
            logging.info(
                f"NEW best cost for parameters in ITERATION: {parameters.iter}; NUM SERVERS: {candidate_solution[0]}; "
                f"BUDGET: {candidate_solution[1]}; PERIOD: {candidate_solution[2]}\n "
                f"resulting in a BEST COST of {int(parameters.best_cost)}")

        else:
            if candidate_cost > parameters.norm_max:
                parameters.norm_max = candidate_cost
            candidate_cost_norm = np.interp(candidate_cost, [1, parameters.norm_max], [0, 200])
            best_cost_norm = np.interp(parameters.best_cost, [1, parameters.norm_max], [0, 200])
            rand_number = np.random.rand()
            factor_prob = np.exp(-(candidate_cost_norm - best_cost_norm) / parameters.curr_temp)

            if rand_number < factor_prob:
                logging.info(f"Candidate with worse solution was accepted with {int(factor_prob * 100)}% chance and "
                             f"RAND VALUE: {rand_number}")
                logging.info(
                    f"Candidate cost: {int(candidate_cost)} and Best cost: {int(parameters.best_cost)} before random "
                    f"acceptance")
                parameters.best_cost = candidate_cost
                parameters.best_solution = candidate_solution
                parameters.best_schedule = tt_schedule

        parameters.curr_temp = def_temp / (1 + parameters.cool * parameters.iter)

    # return the new random changes to have the next candidates
    # we are still going to discuss the boundaries

    number_poll_servers = candidate_solution[0]  # number of poll servers is fixed for the dataset

    period_poll_servers = []
    for i in range(number_poll_servers):
        period = hyperperiod - 1
        while hyperperiod % period != 0 or period < 2:
            period_poll_servers_variation = np.random.randint(-max_period_variation, max_period_variation, size=1)
            period = parameters.best_solution[2][i] + period_poll_servers_variation
            if period == 0:
                period = -1

        period_poll_servers.append(int(period))

    budget_poll_servers = []
    for i in range(number_poll_servers):
        budget = -1
        while budget < 1 or budget > period_poll_servers[i]:
            budget_poll_servers_variation = np.random.randint(-max_budget_variation, max_budget_variation, size=1)
            budget = parameters.best_solution[1][i] + budget_poll_servers_variation
        budget_poll_servers.append(int(budget))

    return budget_poll_servers, period_poll_servers


def task_seperation(t_list):
    """
    Determines the minimum and maximum amount of polling serves to create
    for the given task list
    :param t_list: task list
    :return: number of polling servers
    """
    sep_list = []
    zero_bool = 0
    for task in t_list:
        if task.type == "ET":
            sep_list.append(task.seperation)

    # Find minimum number of polling servers from separation values
    unique_values = list(set(sep_list))
    for value in unique_values:
        if value == 0:
            zero_bool = 1
            break

    min_no_ps = len(unique_values) - zero_bool

    return min_no_ps


def priority_parser(et_tasks):
    """
    reassign the priorities of the et tasks
    :param et_tasks: et tasks list
    :return: updated et tasks list
    """
    # reassignment of the priorities of the ET tasks according to the EDF method -> earliest deadline, biggest priority
    # 1) check the biggest deadline of the ET tasks in the test case
    # 2) divide that by the number of ET tasks in the test case
    # 3) assign priorities based on the space division

    # reassignment of the priorities of the ET tasks according to the lowest duration, biggest priority
    # 1) check the biggest duration of the ET tasks in the test case
    # 2) divide that by the number of possible priorities in the test case
    # 3) assign priorities based on the space division

    big_ct = 0

    for task in et_tasks:
        if big_ct < task.duration:
            big_ct = task.duration

    chunk_size = math.floor(big_ct / 7)
    if big_ct % 7 > 0:
        chunk_size += 1

    for task in et_tasks:
        for i in range(0, 7):
            if task.duration <= chunk_size * (i + 1):
                task.priority = 6 - i
                break

    return et_tasks


def main():
    # create list with an object Task for every task in  the csv files
    logging.debug("Program started, setting initial conditions\n")
    task_list = tasks_parser(testcases_path, test_file)

    # get number of polling servers
    min_no_srv = task_seperation(task_list)  # we decided to use only the min_no_srv

    # get groups of et tasks with same separation number
    first_et_tasks_groups = et_tasks_seperation(task_list, min_no_srv)

    et_tasks_groups = []
    for et_tasks in first_et_tasks_groups:
        et_tasks_groups.append(priority_parser(et_tasks))

    et_wcrt_groups = []
    et_bool_groups = []
    budget_poll_srv = np.array([def_budget for i in range(min_no_srv)])
    period_poll_srv = np.array([def_period for i in range(min_no_srv)])
    poll_srv = create_poll_srv(budget_poll_srv, period_poll_srv)
    num_ps = len(poll_srv)

    # schedule  ET and TT tasks
    tt_schedule, tt_wcrt, tt_hyperperiod = edf_sim(task_list, poll_srv)
    if len(tt_wcrt) == 0:
        tt_schedule_bool = 0
    else:
        tt_schedule_bool = 1
        for i, et_tasks in enumerate(et_tasks_groups):
            supply_mask = supply_poll_server(tt_schedule, i)
            et_bool, et_wcrt = et_schedule(et_tasks, supply_mask)
            et_bool_groups.append(et_bool)
            et_wcrt_groups.append(et_wcrt)
    et_bool = 0
    for bool_var in et_bool_groups:
        if not bool_var:
            et_bool += 1
            break

    # set simulated annealing initial parameters
    cand_sol = [num_ps, [def_budget for i in range(min_no_srv)], [def_period for i in range(min_no_srv)]]
    params = SimAnnealingParams(def_temp, cand_sol, cost_function(tt_wcrt, et_wcrt_groups, et_bool), tt_schedule,
                                def_cooling, 10000)

    logging.info(f"Initial cost: {params.best_cost}")
    logging.debug("Simulated Annealing starting.\n")
    logging.debug(f"Using test case file: {testcases_path}/{test_file}")
    initial_time = time.time()
    curr_time = 0
    # params.curr_temp > 0.1
    while params.curr_temp > 0.01 and curr_time < 120:
        # run simulated annealing
        new_budget, new_period = simulated_annealing(tt_wcrt, et_wcrt_groups, tt_schedule, et_bool, cand_sol,
                                                     params,
                                                     tt_hyperperiod, tt_schedule_bool)

        print(f"Iteration {params.iter} SA with budget of {new_budget} and period of {new_period}")
        print(f"TEMPERATURE: {params.curr_temp}")
        print(f"Best cost: {int(params.best_cost)}")
        print("Best solution:", params.best_solution)

        params.iter = params.iter + 1
        # update parameters
        new_ps = create_poll_srv(new_budget, new_period)
        new_no_ps = len(new_ps)
        cand_sol = [new_no_ps, new_budget, new_period]

        tt_schedule, tt_wcrt, tt_hyperperiod = edf_sim(tasks_parser(testcases_path, test_file), new_ps)

        if len(tt_wcrt) == 0:
            tt_schedule_bool = 0
            continue
        else:
            tt_schedule_bool = 1
        # if TT set is not schedulable , it will not try to do et_schedule
        et_wcrt_groups = []
        et_bool_groups = []
        for i, et_tasks in enumerate(et_tasks_groups):
            supply_mask = supply_poll_server(tt_schedule, i)
            et_bool, et_wcrt = et_schedule(et_tasks, supply_mask)
            et_bool_groups.append(et_bool)
            et_wcrt_groups.append(et_wcrt)
        et_bool = 0
        for bool_var in et_bool_groups:
            if not bool_var:
                et_bool += 1
        curr_time = time.time() - initial_time

    logging.info(
        f"RESULTS: Best Cost of {int(params.best_cost)}\nNo of Servers: {params.best_solution[0]} , "
        f"Budget: {params.best_solution[1]} Period:{params.best_solution[2]}")


if __name__ == "__main__":
    main()
