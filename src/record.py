
NUM_TRIALS = 1

def get_test_record_title(game_name, episode, trainopt, num_tests=NUM_TRIALS, important_sampling=True):
    title = '{}_EPI{}_{}_#Trials{}'.format(game_name, episode, trainopt, num_tests)
    if important_sampling:
        title += '_important_sampling'
    return title

