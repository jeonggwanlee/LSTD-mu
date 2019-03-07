import ipdb

#file_name = '/Users/jklee/temporal_git/LSTD-mu/src/CartPole-v0_EPI1000_initial2_#Test1_important_sampling_KEEP.csv'
file_name = 'CartPole-v0_EPI996_keepBA&notRB_#Trials1.csv'

with open(file_name, 'r') as f:
    lines = [l for l in f]

new_values = []
for line in lines:
    if line[0] == '#':
        continue
    line_truncated = line.split('\n')[0]
    new_values.append(line_truncated)

line = ",".join(new_values)

new_file_name = file_name[:-4] + "removed.csv"

with open(new_file_name, 'w') as f:
    f.write(line)

