import pickle
import ipdb
bin_name = "CartPole-v0_DAN20_PreSoftFalse_ImportantSamplingTrue_FindBestAgentEpi100_best_policy_irl_dan_pickle.bin"

with open(bin_name, 'rb') as rf:
    load = pickle.load(rf)

ipdb.set_trace()
