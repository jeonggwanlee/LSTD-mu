import pickle
import ipdb

dan_opt = False

game_name = "CartPole-v0"
irl_name = "irl"
important_sampling = True

#index_list = ["", "_2", "_3", "_4", "_5"]
#index_list = ["_DEBUG", "_DEBUG2", "_DEBUG_8mul"]
index_list = ["_#Trajs400"]

if dan_opt:
    excel_file = "{}_DAN9_PreSoftFalse_ImportantSampling{}_FindBestAgentEpi30_best_policy_{}_analysis.csv".format(game_name, important_sampling, irl_name)
else:
    excel_file = "{}_RewardBasis9_ImportantSampling{}_FindBestAgentEpi30_best_policy_{}_analysis.csv".format(game_name, important_sampling, irl_name)
    

t_collection_list = []
test_reward_collection_list = []
time_checker_collection_list = []

for index in index_list:

    if dan_opt:
        bin_file = "{}_DAN9_PreSoftFalse_ImportantSampling{}_FindBestAgentEpi30_best_policy_{}_pickle{}.bin".format(game_name, important_sampling, irl_name, index)
    else:
        bin_file = "{}_RewardBasis9_ImportantSampling{}_FindBestAgentEpi30_best_policy_{}_pickle{}.bin".format(game_name, important_sampling, irl_name, index)
    

    with open(bin_file, 'rb') as rf:
        load = pickle.load(rf)
    Best_agents = load[0]
    t_collection = load[1]
    test_reward_collection = load[2]
    if irl_name == "irl_lstdmu" and not dan_opt:
        time_checker_collection = load[3]
    
    t_collection_list.append(t_collection)
    test_reward_collection_list.append(test_reward_collection)
    if irl_name == "irl_lstdmu" and not dan_opt:
        time_checker_collection_list.append(time_checker_collection)

def truncate(number):
    num_trun = "%.4f" % (number)
    return str(num_trun)

with open(excel_file, 'w') as wf:
    for t_collection in t_collection_list:
        t_collection = map(truncate, t_collection)
        line = ','.join(t_collection) + '\n'
        wf.write(line)
    wf.write('\n')

    for test_reward_collection in test_reward_collection_list:
        test_reward_collection = map(truncate, test_reward_collection)
        line = ','.join(test_reward_collection) + '\n'
        wf.write(line)

ipdb.set_trace()
