# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 15:43:40 2020

@author: Prthamesh
"""

import pandas as pd
import random
from copy import deepcopy as dc
import pprint
import time

# function that generates a map
# Inputs: dimension of a map 
def generate_map(dim):
    terrain_type_list = ['Flat','Hill','Forest','Cave'] # list of terrain types
    terrain_type_prob_list = [0.2,0.3,0.3,0.2] # corresponding probabilities of each terrain type
    # map is a dictionary with items of the form (row,col):terrain_type
    mp = {
            (r,c):random.choices(terrain_type_list,terrain_type_prob_list,k=1)[0] 
            for c in range(dim)
            for r in range(dim)
        }
    return mp

# function that displays map as a square grid in a console
# Inputs: map, dimension of a map
def display_map(mp,dim):
    # dictionary of symbols to be displayed for each terrain type
    terrain_type_symbol_dict = {
                                'Flat':'_',
                                'Hill':'^',
                                'Forest':'\u2663',
                                'Cave':'\u2126'
                                }
    print('    ',end='')
    # print column numbers
    for num in range(dim):
        print('{0:^3d}'.format(num),end=' ')
    print()
    print('   ',end='')
    print('-'*((4*dim)+1))
    for r in range(dim):
        print(' {0} |'.format(r),end='') # print row numbers
        for c in range(dim):
            terrain_type = mp[(r,c)]
            terrain_symbol = terrain_type_symbol_dict[terrain_type]
            print(' {0} '.format(terrain_symbol),end='|') # print symbol at position (r,c)
        print()
        print('   ',end='')
        print('-'*((4*dim)+1))
        
# function that sets a target to be hunted
# Inputs: map, dimension of a map
def set_target(mp,dim,given_target_position=None):
    if given_target_position:
        target_position = given_target_position
    else:
        target_r = random.randint(0,dim-1) # row number of target position
        target_c = random.randint(0,dim-1) # column number of target position
        target_position = (target_r,target_c) # target position as a tuple
    
    # nested function that searches for a target in a given cell and returns result of search
    # Inputs: position to be searched as a tuple
    def search(position_to_be_searched):
        # probability that target will not be found in a cell given that target is in a cell
        target_not_found_given_target_in_prob_dict = {'Flat':0.1,'Hill':0.3,'Forest':0.7,'Cave':0.9}
        if position_to_be_searched != target_position: # target is not in the position to be searched
            return False # target is never found in such a position
        else: # target is in the position to be searched
            # terrain type of position to be searched
            terrain_type_of_position_to_be_searched = mp[position_to_be_searched]
            # probability that target won't be found in position to be searched
            # given that target is in that position
            target_not_found_in_position_to_be_searched_prob = target_not_found_given_target_in_prob_dict[terrain_type_of_position_to_be_searched]
            search_result_list = [True,False] # list of possible search results
            # list of probabilities of search results
            search_result_prob_list = [(1-target_not_found_in_position_to_be_searched_prob),
                                       target_not_found_in_position_to_be_searched_prob]
            # perform search and get the search result
            search_result = random.choices(search_result_list,
                                    search_result_prob_list,
                                    k=1)[0]
            return search_result # return the search result
    return search # return search function that hides direct access to the target location
        
# function that generates probabilistic knowledge base
# inputs: map, dimension of a map
def generate_probabilistic_kb(mp,dim):
    target_not_found_given_target_in_prob_dict = {'Flat':0.1,'Hill':0.3,'Forest':0.7,'Cave':0.9}
    pkb  = [[] for _ in range(dim)] # probabilistic knowledge base
    prob_target_in_cell = 1/(dim**2) # initial probability that a cell contains target
    for r in range(dim): # iterate through every row
        for c in range(dim): # iterate through every column
            terrain_type = mp[(r,c)] # terrain type of cell at position (r,c)
            # probability that target will be found in cell
            prob_target_found_in_cell = (prob_target_in_cell * 
                                         (1-target_not_found_given_target_in_prob_dict[terrain_type]))
            # probabilistic knowledge base of a cell at position (r,c)
            cell_pkb = {
                        'prob_target_in_cell':prob_target_in_cell,
                        'prob_target_found_in_cell':prob_target_found_in_cell
                        }
            pkb[r].append(cell_pkb) # append cell's kb to probabilistic kb
    return pkb # return probabilistic knowledge base

# function that displays probabilistic knowledge base as a square grid
# Inputs: probabilistic kb, dimension of a map, name of probability
def display_probabilistic_kb(pkb,dim,prob_name):
    print(prob_name)
    width = 6
    prec = 8
    print('    ',end='')
    for num in range(dim):
        # print column numbers
        print('{0:^{1}d}'.format(num,width),end=' ') 
    print()
    print('   ',end='')
    print('-'*(((width+1)*dim)+1)) 
    for r in range(dim): # iterate through every row
        print(' {0} |'.format(r),end='') # print row numbers
        for c in range(dim): # iterate through every column
            cell_pkb = pkb[r][c]
            print('{0:^{1}.{2}f}'.format(cell_pkb[prob_name],width,prec),end='|')
        print()
        print('   ',end='')
        print('-'*(((width+1)*dim)+1))
      
# function that returns a list of cell positions that have highest probability
# Inputs: probabilistic knowledge base, dimension of a map
# probability name
def get_list_of_highest_prob_cell_positions(pkb,dim,prob_name):
    # list of all probabilities
    list_of_all_prob = [pkb[r][c][prob_name]
                        for r in range(dim)
                        for c in range(dim)]
    max_prob = max(list_of_all_prob) # get maximum probability
    highest_prob_cell_positions_list = [] # list to store positions having maximum probability
    for r in range(dim): # iterate through every row
        for c in range(dim): # iterate through every column
            if pkb[r][c][prob_name] == max_prob: # if cell at (r,c) has maximum probability
                highest_prob_cell_positions_list.append((r,c)) # add position (r,c) to the list
    return highest_prob_cell_positions_list # return list of positions having maximum probability

# function that sets agent in a map
# Inputs: probabilistic knowledge base, dimension of a map, name of agent
def set_agent_in_map(pkb,dim,agent_name):
    if agent_name == 'agent_1': # if basic agent 1
        # all cells have equal target in cell probability
        start_position_r = random.randint(0,dim-1)
        start_position_c = random.randint(0,dim-1)
        # position where agent will begin searching
        start_position = (start_position_r,start_position_c)
    elif agent_name == 'agent_2' or agent_name == 'agent_3' or agent_name == 'agent_4':
        highest_prob_cell_positions_list = get_list_of_highest_prob_cell_positions(pkb,dim,'prob_target_found_in_cell')
        start_position = random.choice(highest_prob_cell_positions_list)
    return start_position

# function that updates probabilistic knowledge base
def update_prob(mp,pkb,dim,searched_position,agent_name):
    target_not_found_given_target_in_prob_dict = {'Flat':0.1,'Hill':0.3,'Forest':0.7,'Cave':0.9}
    pkb_new  = [[{} for c in range(dim)] for r in range(dim)] # new probabilistic knowledge base
    terrain_type_of_searched_position = mp[searched_position]
    searched_r,searched_c = searched_position
    target_in_searched_position_prob = pkb[searched_r][searched_c]['prob_target_in_cell']
    target_not_found_in_searched_position_prob = ((1-target_in_searched_position_prob)*1 + 
            target_in_searched_position_prob * (target_not_found_given_target_in_prob_dict[terrain_type_of_searched_position]))
    for r in range(dim): # iterate through every row
        for c in range(dim): # iterate through every column
            terrain_type = mp[(r,c)]
            target_in_cell_prob = pkb[r][c]['prob_target_in_cell']
            if (r,c) == searched_position:
                numerator = (target_in_cell_prob * 
                             target_not_found_given_target_in_prob_dict[terrain_type])
            else:
                numerator = target_in_cell_prob
            target_in_cell_prob_new = numerator/target_not_found_in_searched_position_prob
            pkb_new[r][c]['prob_target_in_cell'] = target_in_cell_prob_new
            if agent_name != 'agent_1':
                pkb_new[r][c]['prob_target_found_in_cell'] = target_in_cell_prob_new * (1-target_not_found_given_target_in_prob_dict[terrain_type])
    return pkb_new

# function that returns the position in a list nearest to agent's position
def get_nearest_position(agent_position,highest_prob_cell_positions_list):
    nearest_position = highest_prob_cell_positions_list[0]
    nearest_position_distance = (abs(agent_position[0]-nearest_position[0]) + 
                                 abs(agent_position[1]-nearest_position[1]))
    for position in highest_prob_cell_positions_list:
        distance = (abs(agent_position[0]-position[0]) + 
                    abs(agent_position[1]-position[1]))
        if distance < nearest_position_distance:
            nearest_position = position
            nearest_position_distance = distance
    return nearest_position

# function that returns a position to be searched next by basic agent 3
def get_new_position_for_agent_3(pkb,dim,agent_position):
    scores_dict = {}
    for r in range(dim):
        for c in range(dim):
            distance = (abs(r-agent_position[0]) + 
                        abs(c-agent_position[1]))
            target_found_in_cell_prob = pkb[r][c]['prob_target_found_in_cell']
            scores_dict[(r,c)] = ((1+distance) / target_found_in_cell_prob)
    # scores_dict.pop(agent_position)
    min_score = min(scores_dict.values())
    for position in scores_dict:
        if scores_dict[position] == min_score:
            return position

# recursive function that is used by improved agent to perform look-ahead
def get_new_position_rec_func(pkb,dim,agent_position,sequence,depth,cost):
    global min_position
    scores_dict = {}
    for r in range(dim):
        for c in range(dim):
            distance = (abs(r-agent_position[0]) + 
                        abs(c-agent_position[1]))
            target_found_in_cell_prob = pkb[r][c]['prob_target_found_in_cell']
            scores_dict[(r,c)] = ((1+distance) / target_found_in_cell_prob)
    # scores_dict.pop(agent_position)
    min_score = min(scores_dict.values())
    possible_positions_to_move = []
    for position in scores_dict:
        if scores_dict[position] == min_score:
            possible_positions_to_move.append(position)
    if depth == 0:
        position_to_be_searched = possible_positions_to_move[0]
        cost += abs(agent_position[0]-position_to_be_searched[0]) + abs(agent_position[1]-position_to_be_searched[1])
        if cost < min_cost:
            min_position = sequence[0]
    else:
        for searched_position in possible_positions_to_move:
            pkb2 = update_prob(mp,dc(pkb),dim,searched_position,'agent_4')
            sequence2 = dc(sequence)
            sequence2.append(searched_position)
            depth2 = depth - 1
            cost2 = cost + min_score #1 + abs(agent_position[0]-searched_position[0]) + abs(agent_position[1]-searched_position[1])
            get_new_position_rec_func(pkb2,dim,searched_position,sequence2,depth2,cost2)

# function that returns a position to be searched next by improved agent
def get_new_position_for_agent_4(pkb,dim,agent_position,sequence,depth):
    global min_position, min_cost
    min_position = None
    min_cost = 9999999999999999999
    get_new_position_rec_func(pkb,dim,agent_position,sequence,depth,0)
    return min_position
        
# function that moves AI agent by selecting a new position to be searched
def move_agent(pkb,dim,agent_name,does_movement_incur_cost,agent_position):
    if agent_name == 'agent_1': # basic agent 1
        # get list of positions with highest target in cell probability
        highest_prob_cell_positions_list = get_list_of_highest_prob_cell_positions(pkb,dim,'prob_target_in_cell')
        if does_movement_incur_cost: # movement incurs cost
            position_to_be_searched = get_nearest_position(agent_position,highest_prob_cell_positions_list)
        else: # movement does not incur cost
            position_to_be_searched = random.choice(highest_prob_cell_positions_list)
    elif agent_name == 'agent_2': # basic agent 1
        # get list of positions with highest target found in cell probability
        highest_prob_cell_positions_list = get_list_of_highest_prob_cell_positions(pkb,dim,'prob_target_found_in_cell')
        if does_movement_incur_cost: # movement incurs cost
            position_to_be_searched = get_nearest_position(agent_position,highest_prob_cell_positions_list)
        else: # movement does not incur cost
            position_to_be_searched = random.choice(highest_prob_cell_positions_list)
    elif agent_name == 'agent_3': # basic agent 3
        position_to_be_searched = get_new_position_for_agent_3(pkb,dim,agent_position)
    elif agent_name == 'agent_4': # improved agent
        position_to_be_searched = get_new_position_for_agent_4(pkb,dim,agent_position,sequence=[],depth=4)
    return position_to_be_searched # return new position to be searched

# function that executes agent play until target is found
# Inputs: dimension of a map, agent name
def play(dim,agent_name,does_movement_incur_cost,given_target_position=None):
    search = set_target(mp,dim,given_target_position) # set target and search method
    pkb = generate_probabilistic_kb(mp,dim) # get probabilistic knowledge base
    #display_probabilistic_kb(pkb,dim,'prob_target_in_cell')
    #display_probabilistic_kb(pkb,dim,'prob_target_found_in_cell')
    position_to_be_searched = set_agent_in_map(pkb,dim,agent_name) # get initial position to be searched by agent
    is_target_found = False # flag indicating whether the target is found
    total_number_of_actions = 0 # total number of actions required to find the target
    while not is_target_found: # iterate until target is found
        is_target_found = search(position_to_be_searched) # search the position and get the result of search
        position_searched = position_to_be_searched
        total_number_of_actions += 1 # search action
        if is_target_found: # target is found
            #print('Target position:',position_searched)
            #print('Target terrain type:',mp[position_searched])
            #print('Normalization:',check_normalization(pkb,dim,'prob_target_in_cell'))
            return total_number_of_actions,mp[position_searched]
        else: # target is not found
            pkb = update_prob(mp,pkb,dim,position_searched,agent_name) # update probabilistic knowledge base
            #print('Position searched:',position_searched)
            #display_probabilistic_kb(pkb,dim,'prob_target_in_cell')
            #display_probabilistic_kb(pkb,dim,'prob_target_found_in_cell')
            position_to_be_searched = move_agent(pkb,dim,agent_name,does_movement_incur_cost,position_searched) # get new position to be searched
            if does_movement_incur_cost: # if movement incurs cost
                nearest_position_distance = (abs(position_searched[0]-position_to_be_searched[0]) +
                                             abs(position_searched[1]-position_to_be_searched[1]))
                total_number_of_actions += nearest_position_distance # add number of movements to total number of actions
     

total_runs = 100 # Total number of runs to compute average
dim = 20 # dimension of a landscape
counts = {'a1':{'Flat':[0,0],'Hill':[0,0],'Forest':[0,0],'Cave':[0,0]},
          'a2':{'Flat':[0,0],'Hill':[0,0],'Forest':[0,0],'Cave':[0,0]},
          'a3':{'Flat':[0,0],'Hill':[0,0],'Forest':[0,0],'Cave':[0,0]},
          'a4':{'Flat':[0,0],'Hill':[0,0],'Forest':[0,0],'Cave':[0,0]}}
mp = generate_map(dim) # get the map
display_map(mp,dim) # display the map
for run_number in range(total_runs):
    given_target_position = (random.randint(0,dim-1),random.randint(0,dim-1))
    given_target_terrain_type = mp[given_target_position]
    print('Run number',(run_number+1))
    #print('Target position:',given_target_position)
    #print('Target terrain type:',mp[given_target_position])
    a1_num_of_actions,a1_target_terrain_type = play(dim=dim,agent_name='agent_1',does_movement_incur_cost=True,given_target_position=given_target_position)
    a2_num_of_actions,a2_target_terrain_type = play(dim=dim,agent_name='agent_2',does_movement_incur_cost=True,given_target_position=given_target_position)
    a3_num_of_actions,a3_target_terrain_type = play(dim=dim,agent_name='agent_3',does_movement_incur_cost=True,given_target_position=given_target_position)
    a4_num_of_actions,a4_target_terrain_type = play(dim=dim,agent_name='agent_4',does_movement_incur_cost=True,given_target_position=given_target_position)
    counts['a1'][a1_target_terrain_type][0] += a1_num_of_actions
    counts['a2'][a2_target_terrain_type][0] += a2_num_of_actions
    counts['a3'][a3_target_terrain_type][0] += a3_num_of_actions
    counts['a4'][a4_target_terrain_type][0] += a4_num_of_actions
    counts['a1'][a1_target_terrain_type][1] += 1
    counts['a2'][a2_target_terrain_type][1] += 1
    counts['a3'][a3_target_terrain_type][1] += 1
    counts['a4'][a4_target_terrain_type][1] += 1
    
avg_counts = dict(Flat={},Hill={},Forest={},Cave={})
for agent_key in counts:
    for terrain_type in ['Cave','Flat','Hill','Forest']:
        if counts[agent_key][terrain_type][1] != 0:
            avg_counts[terrain_type][agent_key] = (counts[agent_key][terrain_type][0]/counts[agent_key][terrain_type][1])

print('Dimension of a board:',dim)
print('Total runs:',total_runs)
df = pd.DataFrame(avg_counts)
print(df)

avg_counts = dict(a1={},a2={},a3={},a4={})
for agent_key in counts:
    for terrain_type in ['Cave','Flat','Hill','Forest']:
        if counts[agent_key][terrain_type][1] != 0:
            avg_counts[agent_key][terrain_type] = (counts[agent_key][terrain_type][0]/counts[agent_key][terrain_type][1])


























