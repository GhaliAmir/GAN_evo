from src.gans.discriminator_zoo import Discriminator, Discriminator_PReLU, Discriminator_light
from src.gans.generator_zoo import Generator
from src.gans.match_and_train import Arena, GANEnvironment
# from src.new_mongo_interface import save_pure_disc, save_pure_gen, filter_pure_disc, filter_pure_gen
import pickle
import torchvision.datasets as dset
import torchvision.transforms as transforms
from itertools import combinations, product
import torch.optim as optim
import random
from collections import defaultdict
import os
from datetime import datetime, timedelta
import csv
from src.fid_analyser import calc_single_fid_is
from src.mongo_interface import save_pure_disc, save_pure_gen, update_pure_disc, update_pure_gen
from datetime import datetime
from configs import cuda_device
from configs import current_dataset as _dataset
from src.smtp_logger import logger, successfully_completed, smtp_error_bail_out
import smtplib


from src.evo_helpers import hosts_adaptation_check, pathogens_adaptation_check, update_fitnesses, select_best_individuals,\
                            bottleneck_effect, dump_evo, pathogen_sweeps, host_sweeps


#Main Environment, where natural evolution algorithm is implemented
#Environment where evolutionary algorithm take place. 


evo_trace_dump_location = "evolved_hosts_pathogen_map.csv"
evo2_trace_dump_location = "evolved_2_hosts_pathogen_map.csv"
brute_force_trace_dump_location = 'brute_force_pathogen_map.csv'



        
#Implemented to keep track of the advancement of the run in the main
def dump_test(payload_list):
    if not os.path.isfile(test_dump_file):
        open(test_dump_file, 'w')

    with open(test_dump_file, 'a') as destination:
        writer = csv.writer(destination, delimiter='\t')
        writer.writerow(payload_list)

        
def dump_trace(payload_list):
    if not os.path.isfile(trace_dump_file):
        open(trace_dump_file, 'w')

    with open(trace_dump_file, 'a') as destination:
        writer = csv.writer(destination, delimiter='\t')
        writer.writerow(payload_list)


def dump_with_backup(object, location):
    if os.path.isfile(location):
        new_location = location[:-4] + '-bckp-' + datetime.now().isoformat() + location[-4:]
        os.rename(location, new_location)
    pickle.dump(object, open(location, 'wb'))


def render_evolution(random_tags_list):
    for random_tag in random_tags_list:
        print(random_tag, ' <- ', end='')
    print()


def save_pure_disc_helper(discriminator_instance):
    save_pure_disc(discriminator_instance.save_instance_state())


def save_pure_gen_helpler(generator_instance):
    save_pure_gen(generator_instance.save_instance_state())


def update_pure_disc_helper(discriminator_instance):
    result = update_pure_disc(discriminator_instance.random_tag,
                 {'encounter_trace': discriminator_instance.encounter_trace,
                  'self_error': discriminator_instance.real_error,
                  'gen_error_map': discriminator_instance.gen_error_map,
                  'current_fitness': discriminator_instance.current_fitness})
    if result is None:
        save_pure_disc(discriminator_instance.save_instance_state())
        update_pure_disc(discriminator_instance.random_tag,
                 {'encounter_trace': discriminator_instance.encounter_trace,
                  'self_error': discriminator_instance.real_error,
                  'gen_error_map': discriminator_instance.gen_error_map,
                  'current_fitness': discriminator_instance.current_fitness})


def update_pure_gen_helper(generator_instance):
    result = update_pure_gen(generator_instance.random_tag,
                    {'encounter_trace': generator_instance.encounter_trace,
                     'fitness_map': generator_instance.fitness_map})
    if result is None:
        save_pure_gen(generator_instance.save_instance_state())
        update_pure_gen(generator_instance.random_tag,
                    {'encounter_trace': generator_instance.encounter_trace,
                     'fitness_map': generator_instance.fitness_map})

class StopWatch(object):

    def __init__(self):
        self._t = timedelta(microseconds=0)
        self._start = datetime.now()

    def start(self):
        self._start = datetime.now()

    def stop(self):
        delta = datetime.now() - self._start
        self._t += delta
        self._start = datetime.now()

    def get_total_time(self):
        if self._t != 0:
            return self._t.total_seconds() / 60.
        else:
            return 0


def spawn_host_population(individuals_per_species):
    hosts = {
        'base': [],
        'PreLU': [],
        'light': []
    }  # Type: list
    for _ in range(0, individuals_per_species):
        hosts['base'].append(Discriminator(ngpu=environment.ngpu,
                         latent_vector_size=environment.latent_vector_size,
                         discriminator_latent_maps=64,
                         number_of_colors=environment.number_of_colors).to(environment.device))
        hosts['PreLU'].append(Discriminator_PReLU(ngpu=environment.ngpu,
                         latent_vector_size=environment.latent_vector_size,
                         discriminator_latent_maps=64,
                         number_of_colors=environment.number_of_colors).to(environment.device))
        hosts['light'].append(Discriminator(ngpu=environment.ngpu,
                                         latent_vector_size=environment.latent_vector_size,
                                         discriminator_latent_maps=32,
                                         number_of_colors=environment.number_of_colors).to(environment.device))

    for host_type, _hosts in hosts.items():
        print(host_type, ': ', [host.random_tag for host in _hosts])

    return hosts


def spawn_pathogen_population(starting_cluster):
    pathogens = []
    for _ in range(0, starting_cluster):
        pathogens.append(Generator(ngpu=environment.ngpu,
                    latent_vector_size=environment.latent_vector_size,
                    generator_latent_maps=64,
                    number_of_colors=environment.number_of_colors).to(environment.device))

    print('pathogens: ', [pathogen.random_tag for pathogen in pathogens])
    return pathogens


def cross_train_iteration(hosts, pathogens, host_type_selector, epochs=1, timer=None):

    dump_trace(['>>>', 'cross-train',
                [host.random_tag for host in hosts[host_type_selector]],
                [pathogen.random_tag for pathogen in pathogens],
                host_type_selector, epochs,
                datetime.now().isoformat()])

    print('cross-training round with host type: %s' % host_type_selector)

    for (host_no, host), (pathogen_no, pathogen) in product(enumerate(hosts[host_type_selector]),
                                                            enumerate(pathogens)):

        arena = Arena(environment=environment,
                  generator_instance=pathogen,
                  discriminator_instance=host,
                  generator_optimizer_partial=gen_opt_part,
                  discriminator_optimizer_partial=disc_opt_part)

        dump_trace(['pre-train:',
                    host_no, pathogen_no,
                    arena.discriminator_instance.random_tag,
                    arena.generator_instance.random_tag, ])
        
        
        #EVO -- added here to save correct state of each parent in cross_train(), needed before going into evolve in pop
        #pathogens_adaptation_check(pathogen)
        #hosts_adaptation_check(host)
        
        
        arena.cross_train(epochs, timer=timer, commit=False)
        #EVO -- a fresh new generation starts its journey from here (with many attribute values reset --not the fitness)
                
        
        #EVO -- debug
        dump_test(['1/2 - INITIALLY INSIDE CROSS_TRAIN_ITERATION'])
        dump_test(['DISCRIMINATOR INDEX:', host_no, 'WITH RANDOM TAG:', host.random_tag, 'WIN RATE:', host.win_rate,\
                   'CURRENT FITNESS:', host.current_fitness, 'SKILL RATING TABLE:', host.skill_rating_games,\
                   'AND GEN ERROR MAP:', host.gen_error_map,\
                   'TAG TRACE:', host.tag_trace, 'UNIQUE KEY:', host.key])
        
        dump_test(['GENERATOR INDEX: ', pathogen_no, ' WITH RANDOM TAG: ', pathogen.random_tag, ' WIN RATE: ', pathogen.win_rate,\
                   ' CURRENT FITNESS: ', pathogen.current_fitness, ' SKILL RATING TABLE: ', pathogen.skill_rating_games,\
                   ' AND GEN FITNESS MAP: ', pathogen.fitness_map,\
                    'TAG TRACE: ', pathogen.tag_trace, 'UNIQUE KEY:', pathogen.key])
        
        dump_test([''])
        
        arena.sample_images()

        current_fid, current_is = calc_single_fid_is(arena.generator_instance.random_tag)
        dump_trace(['sampled images from',
                    pathogen_no,
                    arena.generator_instance.random_tag, current_fid, current_is, arena.generator_instance.current_fitness,\
                    arena.generator_instance.state])

        #EVOOOOOOOOOOOO
        dump_trace(['against host',
                    host_no,
                    arena.discriminator_instance.random_tag, arena.discriminator_instance.current_fitness,\
                    arena.discriminator_instance.state])
        
        
        
        arena_match_results = arena.match(commit=False)
        #here the skill rating tables are ready --> need to compute and update fitness next


        dump_trace(['post-cross-train and match:',
                    host_no, pathogen_no,
                    arena.discriminator_instance.random_tag,
                    arena.generator_instance.random_tag,
                    arena_match_results[0], arena_match_results[1],
                    arena.discriminator_instance.current_fitness,
                    arena.generator_instance.current_fitness])

        
        print("%s: real_err: %s, gen_err: %s" % (
            arena.generator_instance.random_tag,
            arena_match_results[0], arena_match_results[1]))        
        
    #EVO
    update_fitnesses(hosts[host_type_selector])
    update_fitnesses(pathogens)
    
    
    for (host_no, host), (pathogen_no, pathogen) in product(enumerate(hosts[host_type_selector]),
                                                            enumerate(pathogens)):

        arena = Arena(environment=environment,
                  generator_instance=pathogen,
                  discriminator_instance=host,
                  generator_optimizer_partial=gen_opt_part,
                  discriminator_optimizer_partial=disc_opt_part)
        
        
        arena_match_results = arena.match(timer=timer, commit=False)


        dump_trace(['final cross-match:',
                    host_no, pathogen_no,
                    arena.discriminator_instance.random_tag,
                    arena.generator_instance.random_tag,
                    arena_match_results[0], arena_match_results[1],
                    arena.discriminator_instance.current_fitness,
                    arena.generator_instance.current_fitness])
                    
        
        print("%s vs %s: real_err: %s, gen_err: %s" % (
            arena.generator_instance.random_tag,
            arena.discriminator_instance.random_tag,
            arena_match_results[0], arena_match_results[1]))
    
    
    #EVO
    update_fitnesses(hosts[host_type_selector])
    update_fitnesses(pathogens)
    
    #EVO -- tested method --works properly (unused until now)
    #best_hosts = select_best_individuals(hosts[host_type_selector])
    #best_pathogens = select_best_individuals(pathogens)
        
    
    
    #EVO -- debug/test loop only
    dump_test(['2/2 - AFTER LAST UPDATE IN CROSS_TRAIN_ITERATION'])
    
    for host in hosts[host_type_selector]:        
        dump_test(['DISCRIMINATOR WITH RANDOM TAG: ', host.random_tag, ' WIN RATE: ', host.win_rate,\
                   ' CURRENT FITNESS: ', host.current_fitness, ' SKILL RATING TABLE: ', host.skill_rating_games,\
                   ' AND GEN ERROR MAP: ', host.gen_error_map,\
                    'TAG TRACE: ', host.tag_trace, 'UNIQUE KEY:', host.key])
    
    for pathogen in pathogens:       
        dump_test(['GENERATOR INDEX WITH RANDOM TAG: ', pathogen.random_tag, ' WIN RATE: ', pathogen.win_rate,\
                   ' CURRENT FITNESS: ', pathogen.current_fitness, ' SKILL RATING TABLE: ', pathogen.skill_rating_games,\
                   ' AND GEN FITNESS MAP: ', pathogen.fitness_map,\
                    'TAG TRACE: ', pathogen.tag_trace, 'UNIQUE KEY:', pathogen.key])
    
    
    dump_test([''])
    dump_test([''])
    
    for host in hosts[host_type_selector]:
        print('host', host.random_tag, host.current_fitness, host.gen_error_map)
        save_pure_disc_helper(host)

    for pathogen in pathogens:
        print('pathogen', pathogen.random_tag, pathogen.fitness_map)
        render_evolution(pathogen.tag_trace)
        save_pure_gen_helpler(pathogen)

    dump_trace(['<<<', 'cross-train',
                datetime.now().isoformat()])


def round_robin_iteration(hosts, pathogens, host_type_selector, epochs=1,
                          rounds=None, randomized=False, timer=None):

    dump_trace(['>>>', 'round-robin',
                [host.random_tag for host in hosts[host_type_selector]],
                [pathogen.random_tag for pathogen in pathogens],
                host_type_selector, epochs,
                datetime.now().isoformat()])

    if rounds is None:
        rounds = len(hosts) * len(pathogens)

    print('round-robin round with host type: %s' % host_type_selector)

    if randomized:
        hosts_no = random.choices(range(len(hosts[host_type_selector])), k=rounds)
        pathogens_no = random.choices(range(len(pathogens)), k=rounds)

        generator = zip(hosts_no, pathogens_no)

    else:
        generator_list = [(host_no, pathogen_no) for host_no, pathogen_no in
                          product(range(len(hosts[host_type_selector])), range(len(pathogens)))]
        # generator = product(range(len(hosts[host_type_selector])), range(len(pathogens)))
        repeats = int(rounds/len(generator_list))
        generator = generator_list*repeats

    
    dump_test(['1/2 - INITIALLY IN ROUND_ROBIN_ITERATION'])
    
    for host in hosts[host_type_selector]:        
        dump_test(['DISCRIMINATOR WITH RANDOM TAG: ', host.random_tag, ' WIN RATE: ', host.win_rate,\
                   ' CURRENT FITNESS: ', host.current_fitness, ' SKILL RATING TABLE: ', host.skill_rating_games,\
                   ' AND GEN ERROR MAP: ', host.gen_error_map,\
                    'TAG TRACE: ', host.tag_trace, 'UNIQUE KEY:', host.key])
    
    for pathogen in pathogens:       
        dump_test(['GENERATOR INDEX WITH RANDOM TAG: ', pathogen.random_tag, ' WIN RATE: ', pathogen.win_rate,\
                   ' CURRENT FITNESS: ', pathogen.current_fitness, ' SKILL RATING TABLE: ', pathogen.skill_rating_games,\
                   ' AND GEN FITNESS MAP: ', pathogen.fitness_map,\
                    'TAG TRACE: ', pathogen.tag_trace, 'UNIQUE KEY:', pathogen.key])
    
    dump_test([''])
    
    for host_no, pathogen_no in generator:

        host = hosts[host_type_selector][host_no]
        pathogen = pathogens[pathogen_no]

        arena = Arena(environment=environment,
                  generator_instance=pathogen,
                  discriminator_instance=host,
                  generator_optimizer_partial=gen_opt_part,
                  discriminator_optimizer_partial=disc_opt_part)

        dump_trace(['pre-train:',
                    host_no, pathogen_no,
                    arena.discriminator_instance.random_tag,
                    arena.generator_instance.random_tag])

        arena.cross_train(epochs, timer=timer, commit=False)

        arena.sample_images()

        current_fid, current_is = calc_single_fid_is(arena.generator_instance.random_tag)
        dump_trace(['sampled images from',
                    pathogen_no,
                    arena.generator_instance.random_tag, current_fid, current_is, arena.generator_instance.current_fitness,\
                    arena.generator_instance.state])

        
        #EVOOOOOOOOOOOO
        dump_trace(['against host',
                    host_no,
                    arena.discriminator_instance.random_tag, arena.discriminator_instance.current_fitness,\
                    arena.discriminator_instance.state])
        
        
        
        arena_match_results = arena.match(timer=timer, commit=False)

        dump_trace(['post-cross-train and match:',
                    host_no, pathogen_no,
                    arena.discriminator_instance.random_tag,
                    arena.generator_instance.random_tag,
                    arena_match_results[0], arena_match_results[1],
                    arena.discriminator_instance.current_fitness,
                    arena.generator_instance.current_fitness])
        
                    #arena.generator_instance.fitness_map.get(
                    #    arena.discriminator_instance.random_tag, 0.05)])

        print("%s: real_err: %s, gen_err: %s" % (
            arena.generator_instance.random_tag,
            arena_match_results[0], arena_match_results[1]))

        
    #EVO
    update_fitnesses(hosts[host_type_selector])
    update_fitnesses(pathogens)
    
    
    for (host_no, host), (pathogen_no, pathogen) in product(enumerate(hosts[host_type_selector]),
                                                            enumerate(pathogens)):

        arena = Arena(environment=environment,
                  generator_instance=pathogen,
                  discriminator_instance=host,
                  generator_optimizer_partial=gen_opt_part,
                  discriminator_optimizer_partial=disc_opt_part)

        arena_match_results = arena.match(timer=timer, commit=False)

        dump_trace(['final cross-match:',
                    host_no, pathogen_no,
                    arena.discriminator_instance.random_tag,
                    arena.generator_instance.random_tag,
                    arena_match_results[0], arena_match_results[1],
                    arena.discriminator_instance.current_fitness,
                    arena.generator_instance.current_fitness])
                    
                    #arena.generator_instance.fitness_map.get(
                    #    arena.discriminator_instance.random_tag, 0.05)])

        print("%s vs %s: real_err: %s, gen_err: %s" % (
            arena.generator_instance.random_tag,
            arena.discriminator_instance.random_tag,
            arena_match_results[0], arena_match_results[1]))

    
    #EVO
    update_fitnesses(hosts[host_type_selector])
    update_fitnesses(pathogens)
    
    
    dump_test(['2/2 - AFTER LASR UPDATE IN ROUND_ROBIN_ITERATION'])
    
    for host in hosts[host_type_selector]:        
        dump_test(['DISCRIMINATOR WITH RANDOM TAG: ', host.random_tag, ' WIN RATE: ', host.win_rate,\
                   ' CURRENT FITNESS: ', host.current_fitness, ' SKILL RATING TABLE: ', host.skill_rating_games,\
                   ' AND GEN ERROR MAP: ', host.gen_error_map,\
                    'TAG TRACE: ', host.tag_trace, 'UNIQUE KEY:', host.key])
    
    for pathogen in pathogens:       
        dump_test(['GENERATOR INDEX WITH RANDOM TAG: ', pathogen.random_tag, ' WIN RATE: ', pathogen.win_rate,\
                   ' CURRENT FITNESS: ', pathogen.current_fitness, ' SKILL RATING TABLE: ', pathogen.skill_rating_games,\
                   ' AND GEN FITNESS MAP: ', pathogen.fitness_map,\
                    'TAG TRACE: ', pathogen.tag_trace, 'UNIQUE KEY:', pathogen.key])
    
    dump_test([''])
    dump_test([''])
    
    for host in hosts[host_type_selector]:
        print('host', host.random_tag, host.current_fitness, host.gen_error_map)
        save_pure_disc_helper(host)

    for pathogen in pathogens:
        print('pathogen', pathogen.random_tag, pathogen.fitness_map)
        save_pure_gen_helpler(pathogen)
        render_evolution(pathogen.tag_trace)

    dump_trace(['<<<', 'round-robin',
                datetime.now().isoformat()])


def round_robin_deterministic(individuals_per_species, starting_cluster):

    dump_trace(['>>', 'deterministic base round robin 2', individuals_per_species, starting_cluster,
                datetime.now().isoformat()])

    hosts = spawn_host_population(individuals_per_species)
    pathogens = spawn_pathogen_population(starting_cluster)

    timer = StopWatch()

    round_robin_iteration(hosts, pathogens, 'base',
                          1,
                          rounds=3*individuals_per_species*starting_cluster,
                          randomized=False, timer=timer)

    host_map = {}
    pathogen_map = {}
    for host in hosts['base']:
        host_map[host.random_tag] = [host.gen_error_map, host.current_fitness, host.real_error,
                                     host.tag_trace]

    for pathogen in pathogens:
        pathogen_map[pathogen.random_tag] = [pathogen.fitness_map, pathogen.tag_trace]

    dump_with_backup((host_map, pathogen_map), evo_trace_dump_location)
    # pickle.dump((host_map, pathogen_map), open('evolved_hosts_pathogen_map.dmp', 'wb'))

    dump_trace(['<<', 'deterministic base round robin 2', datetime.now().isoformat(),
                timer.get_total_time()])


def round_robin_randomized(individuals_per_species, starting_cluster):

    dump_trace(['>>', 'stochastic base round robin', individuals_per_species, starting_cluster,
                datetime.now().isoformat()])

    hosts = spawn_host_population(individuals_per_species)
    pathogens = spawn_pathogen_population(starting_cluster)

    timer = StopWatch()

    cross_train_iteration(hosts, pathogens, 'base', 1, timer=timer)
    round_robin_iteration(hosts, pathogens, 'base',
                          1,
                          rounds=2*individuals_per_species*starting_cluster,
                          randomized=True,
                          timer=timer)

    host_map = {}
    pathogen_map = {}
    for host in hosts['base']:
        host_map[host.random_tag] = [host.gen_error_map, host.current_fitness, host.real_error,
                                     host.tag_trace]

    for pathogen in pathogens:
        pathogen_map[pathogen.random_tag] = [pathogen.fitness_map, pathogen.tag_trace]

    dump_with_backup((host_map, pathogen_map), evo_trace_dump_location)
    # pickle.dump((host_map, pathogen_map), open('evolved_hosts_pathogen_map.dmp', 'wb'))

    dump_trace(['<<', 'stochastic base round robin', datetime.now().isoformat(),
                timer.get_total_time()])


def chain_progression(individuals_per_species, starting_cluster):

    dump_trace(['>>', 'chain progression', individuals_per_species, starting_cluster,
                datetime.now().isoformat()])

    hosts = spawn_host_population(individuals_per_species)
    pathogens = spawn_pathogen_population(starting_cluster)

    timer = StopWatch()

    cross_train_iteration(hosts, pathogens, 'light', 1, timer=timer)
    cross_train_iteration(hosts, pathogens, 'PreLU', 1, timer=timer)
    cross_train_iteration(hosts, pathogens, 'base', 3, timer=timer)

    host_map = {}
    pathogen_map = {}
    for host in hosts['base']:
        host_map[host.random_tag] = [host.gen_error_map, host.current_fitness, host.real_error,
                                     host.tag_trace]

    for pathogen in pathogens:
        pathogen_map[pathogen.random_tag] = [pathogen.fitness_map, pathogen.tag_trace]

    dump_with_backup((host_map, pathogen_map), evo_trace_dump_location)
    # pickle.dump((host_map, pathogen_map), open('evolved_hosts_pathogen_map.dmp', 'wb'))

    dump_trace(['<<', 'chain progression', datetime.now().isoformat(),
                timer.get_total_time()])


def homogenus_chain_progression(individuals_per_species, starting_cluster):

    dump_trace(['>>', 'homogenous chain progression', individuals_per_species, starting_cluster,
                datetime.now().isoformat()])

    hosts = spawn_host_population(individuals_per_species)
    pathogens = spawn_pathogen_population(starting_cluster)

    timer = StopWatch()

    cross_train_iteration(hosts, pathogens, 'base', 1, timer=timer)
    cross_train_iteration(hosts, pathogens, 'base', 1, timer=timer)
    cross_train_iteration(hosts, pathogens, 'base', 3, timer=timer)

    host_map = {}
    pathogen_map = {}
    for host in hosts['base']:
        host_map[host.random_tag] = [host.gen_error_map, host.current_fitness, host.real_error,
                                     host.tag_trace]

    for pathogen in pathogens:
        pathogen_map[pathogen.random_tag] = [pathogen.fitness_map, pathogen.tag_trace]

    dump_with_backup((host_map, pathogen_map), evo_trace_dump_location)
    # pickle.dump((host_map, pathogen_map), open('evolved_hosts_pathogen_map.dmp', 'wb'))

    dump_trace(['<<', 'homogenous chain progression', datetime.now().isoformat(),
                timer.get_total_time()])


# CURRENTPASS: [complexity] cyclomatic complexity=17
def evolve_in_population(hosts_list, pathogens_list, pathogen_epochs_budget, fit_reset=False,
                         timer=None):

        
    
    dump_trace(['>>>', 'evolve_in_population',
                [host.random_tag for host in hosts_list],
                [pathogen.random_tag for pathogen in pathogens_list],
                pathogen_epochs_budget,
                datetime.now().isoformat()])

    iterations_limiter = pathogen_epochs_budget * 5

    pathogens_index = list(range(0, len(pathogens_list)))
    hosts_index = list(range(0, len(hosts_list)))

    if fit_reset:
        #EVO
        pathogens_fitnesses = [1500.]*len(pathogens_list)
        hosts_fitnesses = [1500.]*len(hosts_list)
        
        #add code to really reset the fitnesses here (reset only the current_fitness attribute?)
        for host in hosts_list:
            host.current_fitness = 1500
            
        for pathogen in pathogens_list:
            pathogen.current_fitness = 1500
        
    else:
        
        #EVO
        #pathogens_fitnesses = [pathogen_fitness_retriever(_pathogen) for _pathogen in pathogens_list] 
        pathogens_fitnesses = [_pathogen.current_fitness for _pathogen in pathogens_list]
        hosts_fitnesses = [_host.current_fitness for _host in hosts_list]

    
    host_idx_2_pathogens_carried = defaultdict(list)

    i = 0

    while i < pathogen_epochs_budget:
        if pathogen_epochs_budget < 0:
            dump_trace(['iterations budget overflow break'])
            break
        pathogen_epochs_budget -= 1

        print('current fitness tables: %s; %s' % (hosts_fitnesses, pathogens_fitnesses))
        dump_trace(['current tag/fitness tables',
                    [host.random_tag for host in hosts_list],
                    [pathogen.random_tag for pathogen in pathogens_list],
                    hosts_fitnesses, pathogens_fitnesses])

        # TODO: add a restart from a static state

        # TODO: realistically, we need to look in the carrying tables and then attempt to cross
        #  the infections to other, more efficient hosts.

        
        #EVO -- the only place where we actually use the fitness values, as weights to choose who to train against whom.
        current_host_idx = random.choices(hosts_index, weights=hosts_fitnesses)[0]
        current_pathogen_idx = random.choices(pathogens_index, weights=pathogens_fitnesses)[0]

        arena = Arena(environment=environment,
                  generator_instance=pathogens_list[current_pathogen_idx],
                  discriminator_instance=hosts_list[current_host_idx],
                  generator_optimizer_partial=gen_opt_part,
                  discriminator_optimizer_partial=disc_opt_part)
        
        
        
        arena_match_results = arena.match(timer=timer, commit=False)
        
        
        update_fitnesses([hosts_list[current_host_idx]])
        update_fitnesses([pathogens_list[current_pathogen_idx]])
        
        #EVO
        hosts_fitnesses[current_host_idx] = hosts_list[current_host_idx].current_fitness
        pathogens_fitnesses[current_pathogen_idx] = pathogens_list[current_pathogen_idx].current_fitness
    
        
        
        #EVO -- debug
        dump_test(['1/3 - INSIDE EVOLVE IN POP, AFTER FIRST SKILL RATING UPDATE'])
        dump_test(['DISCRIMINATOR INDEX: ', current_host_idx, ' WITH RANDOM TAG: ', hosts_list[current_host_idx].random_tag,\
                    ' WIN RATE: ', hosts_list[current_host_idx].win_rate, ' CURRENT FITNESS:',\
                    hosts_list[current_host_idx].current_fitness, ' SKILL RATING TABLE: ',\
                    hosts_list[current_host_idx].skill_rating_games,' AND GEN ERROR MAP: ',\
                    hosts_list[current_host_idx].gen_error_map,\
                    'TAG TRACE: ', hosts_list[current_host_idx].tag_trace, 'UNIQUE KEY:', hosts_list[current_host_idx].key])
        
        dump_test(['GENERATOR INDEX: ', current_pathogen_idx, ' WITH RANDOM TAG: ', pathogens_list[current_pathogen_idx].random_tag,\
                    ' WIN RATE: ', pathogens_list[current_pathogen_idx].win_rate,' CURRENT FITNESS: ',\
                    pathogens_list[current_pathogen_idx].current_fitness, ' SKILL RATING TABLE: ',\
                    pathogens_list[current_pathogen_idx].skill_rating_games,' AND GEN FITNESS MAP: ',\
                    pathogens_list[current_pathogen_idx].fitness_map,\
                    'TAG TRACE: ', pathogens_list[current_pathogen_idx].tag_trace,\
                    pathogens_list[current_pathogen_idx].key])
        
        dump_test([''])
        
        print("%s: real_err: %s, gen_err: %s; updated fitnesses: host: %s path: %s" % (
            arena.generator_instance.random_tag,
            arena_match_results[0], arena_match_results[1],
            arena.discriminator_instance.current_fitness,
            arena.generator_instance.current_fitness))
            

        dump_trace(['infection attempt:',
                    current_host_idx, current_pathogen_idx,
                    arena.discriminator_instance.random_tag,
                    arena.generator_instance.random_tag, arena_match_results[0], arena_match_results[1],
                    arena.discriminator_instance.current_fitness,
                    arena.generator_instance.current_fitness])
                    

        
        #EVO
        if arena.generator_instance.current_fitness > 1000 and\
            arena.generator_instance.current_fitness + 500 > arena.discriminator_instance.current_fitness:
            #For an infection to take place we need a minimum pathogen fitness of 1000 (otherwise considered too weak),
            #and a difference in fitness between pathogen and host, for the latter to be infected, of maximum 500.
            
            
            #infection
            if current_pathogen_idx not in host_idx_2_pathogens_carried[current_host_idx]:

                host_idx_2_pathogens_carried[current_host_idx].append(current_pathogen_idx)
                #print('debug: host-pathogen mapping:', host_idx_2_pathogens_carried[
                    #current_host_idx], current_host_idx, current_pathogen_idx)

            dump_trace(['infection successful, current host state:',
                        host_idx_2_pathogens_carried[current_host_idx],
                        arena.discriminator_instance.gen_error_map,
                        current_host_idx,
                        arena.discriminator_instance.current_fitness])

            #EVO
            if arena.discriminator_instance.current_fitness > arena.generator_instance.current_fitness + 150:
            #We consider the infection silent if the difference in fitness is more than 150, otherwise (if less) full.
            
                
                #EVO
                arena.generator_instance.coadaptation = True
                arena.discriminator_instance.coadaptation = True
                
                pathogens_adaptation_check(arena.generator_instance)
                hosts_adaptation_check(arena.discriminator_instance)
                
                pathogen_sweeps(arena.generator_instance)
                host_sweeps(arena.discriminator_instance)
                
                
                #immune system is not bothered
                # print('debug: pop evolve: silent infection')
                dump_trace(['silent infection'])
                arena.cross_train(gen_only=True, timer=timer, commit=False)#EVO -- used to train both with previous conditions
                i += 0.5
            else:
                
                #EVO
                arena.generator_instance.coadaptation = False
                arena.discriminator_instance.coadaptation = False
                
                pathogens_adaptation_check(arena.generator_instance)
                hosts_adaptation_check(arena.discriminator_instance)
                
                pathogen_sweeps(arena.generator_instance)
                host_sweeps(arena.discriminator_instance)
                
                
                #immune sytem is active and competitive evolution happens:
                # print('debug: pop evolve: full infection')
                dump_trace(['full infection'])
                arena.cross_train(timer=timer, commit=False)
                i += 1

            arena.sample_images()
            current_fid, current_is = calc_single_fid_is(arena.generator_instance.random_tag)

            dump_trace(['sampled images from', current_pathogen_idx,
                        arena.generator_instance.random_tag, current_fid, current_is, arena.generator_instance.current_fitness,\
                        arena.generator_instance.state])
            
            
            
            #EVOOOOOOOOOOOO
            dump_trace(['against host',
                        current_host_idx,
                        arena.discriminator_instance.random_tag, arena.discriminator_instance.current_fitness,\
                        arena.discriminator_instance.state])
                        
            
            
            
            arena_match_results = arena.match(timer=timer, commit=False)
            
            
            #EVO
            update_fitnesses([hosts_list[current_host_idx]])
            update_fitnesses([pathogens_list[current_pathogen_idx]])
                        
            
            dump_trace(['post-infection',
                        current_host_idx, current_pathogen_idx,
                        arena.discriminator_instance.random_tag,
                        arena.generator_instance.random_tag,
                        arena_match_results[0], arena_match_results[1],
                        arena.discriminator_instance.current_fitness,
                        arena.generator_instance.current_fitness])

            # print("debug: pop evolve: post-train: real_err: %s, gen_err: %s" % (
                # arena_match_results[0], arena_match_results[1]))
            # print("debug: pop evolve: fitness map", arena.generator_instance.fitness_map)
            # print("debug: pop evolve: updated fitnesses: host: %s path: %s" % (
            #     arena.discriminator_instance.current_fitness,
            #     arena.generator_instance.fitness_map.get(
            #     arena.discriminator_instance.random_tag, 0.05)))

            
            #EVO
            hosts_fitnesses[current_host_idx] = arena.discriminator_instance.current_fitness
            pathogens_fitnesses[current_pathogen_idx] = arena.generator_instance.current_fitness
        
        
        else: #NO INFECTION
            
            # print('debug: pop evolve: infection fails')

            if current_pathogen_idx in host_idx_2_pathogens_carried[current_host_idx]:
                host_idx_2_pathogens_carried[current_host_idx].remove(current_pathogen_idx)

            
            #EVO -- check the results here (how are the maps)
            
            #no need to update these (not related here)            
            #arena.generator_instance.silent_adaptation = False
            #arena.discriminator_instance.silent_adaptation = False
            
            arena.discriminator_instance.gen_error_map.pop(arena.generator_instance.key, None)
            
            pathogens_adaptation_check(arena.generator_instance)
            hosts_adaptation_check(arena.discriminator_instance)

            pathogen_sweeps(arena.generator_instance)
            host_sweeps(arena.discriminator_instance)
            
            
#             #instead of the one we excluded for consistency reasons
#             update_fitnesses([hosts_list[current_host_idx]])
#             update_fitnesses([pathogens_list[current_pathogen_idx]])
            
            
            #EVO
            hosts_fitnesses[current_host_idx] = arena.discriminator_instance.current_fitness
            pathogens_fitnesses[current_pathogen_idx] = arena.generator_instance.current_fitness


            
            
            dump_trace(['infection failed, current host state:',
                        host_idx_2_pathogens_carried[current_host_idx],
                        arena.discriminator_instance.gen_error_map,
                        current_host_idx,
                        arena.discriminator_instance.current_fitness])

    
    
    encountered_pathogens = []
    for (host_no, host), (pathogen_no, pathogen) in product(enumerate(hosts_list),
                                                            enumerate(pathogens_list)):

        arena = Arena(environment=environment,
                  generator_instance=pathogen,
                  discriminator_instance=host,
                  generator_optimizer_partial=gen_opt_part,
                  discriminator_optimizer_partial=disc_opt_part)
        
        
        arena_match_results = arena.match(timer=timer, commit=False)
        
        
        
        if pathogen not in encountered_pathogens:

            arena.sample_images()
            current_fid, current_is = calc_single_fid_is(arena.generator_instance.random_tag)

            dump_trace(['sampled images from',
                        pathogen_no,
                        arena.generator_instance.random_tag,
                        current_fid, current_is, arena.generator_instance.current_fitness, arena.generator_instance.state])

            
            
            #EVOOOOOOOOOOOO
            dump_trace(['against host',
                        host_no,
                        arena.discriminator_instance.random_tag, arena.discriminator_instance.current_fitness,\
                        arena.discriminator_instance.state])
            
            
            
            
            encountered_pathogens.append(pathogen)

        dump_trace(['final cross-match:',
                    host_no, pathogen_no,
                    arena.discriminator_instance.random_tag,
                    arena.generator_instance.random_tag,
                    arena_match_results[0], arena_match_results[1],
                    arena.discriminator_instance.current_fitness,
                    arena.generator_instance.current_fitness])

    
    
    #EVO -- debug loop
    dump_test(['2/3 - AFTER THE WHOLE INFECTION PROCESS AND ITS FINAL CROSS MATCH'])
    for (host_no, host) in enumerate(hosts_list):
        dump_test(['DISCRIMINATOR INDEX: ', host_no, ' WITH RANDOM TAG: ', host.random_tag, ' WIN RATE: ', host.win_rate,\
                   ' CURRENT FITNESS: ', host.current_fitness, ' SKILL RATING TABLE: ', host.skill_rating_games,\
                   ' AND GEN ERROR MAP: ', host.gen_error_map,\
                    'TAG TRACE: ', host.tag_trace, 'UNIQUE KEY:', host.key])
        
    for (pathogen_no, pathogen) in enumerate(pathogens_list):
        dump_test(['GENERATOR INDEX: ', pathogen_no, ' WITH RANDOM TAG: ', pathogen.random_tag, ' WIN RATE: ', pathogen.win_rate,\
                   ' CURRENT FITNESS: ', pathogen.current_fitness, ' SKILL RATING TABLE: ', pathogen.skill_rating_games,\
                   ' AND GEN FITNESS MAP: ', pathogen.fitness_map,\
                    'TAG TRACE: ', pathogen.tag_trace, 'UNIQUE KEY:', pathogen.key])
                    

    dump_test([''])
    
    #EVO
    update_fitnesses(hosts_list)
    update_fitnesses(pathogens_list)
    
    pathogens_fitnesses = [_pathogen.current_fitness for _pathogen in pathogens_list]
    hosts_fitnesses = [_host.current_fitness for _host in hosts_list]
    
    
    #EVO -- debug loop
    dump_test(['3/3 - AFTER LAST UPDATE IN EVOLVE IN POP'])
    
    for (host_no, host) in enumerate(hosts_list):
        dump_test(['DISCRIMINATOR INDEX: ', host_no, ' WITH RANDOM TAG: ', host.random_tag, ' WIN RATE: ', host.win_rate,\
                   ' CURRENT FITNESS: ', host.current_fitness, ' SKILL RATING TABLE: ', host.skill_rating_games,\
                   ' AND GEN ERROR MAP: ', host.gen_error_map,\
                    'TAG TRACE: ', host.tag_trace, 'UNIQUE KEY:', host.key])
        
    for (pathogen_no, pathogen) in enumerate(pathogens_list):
        dump_test(['GENERATOR INDEX: ', pathogen_no, ' WITH RANDOM TAG: ', pathogen.random_tag, ' WIN RATE: ', pathogen.win_rate,\
                   ' CURRENT FITNESS: ', pathogen.current_fitness, ' SKILL RATING TABLE: ', pathogen.skill_rating_games,\
                   ' AND GEN FITNESS MAP: ', pathogen.fitness_map,\
                    'TAG TRACE: ', pathogen.tag_trace, 'UNIQUE KEY:', pathogen.key])
    
    dump_test([''])
    dump_test([''])
    
    
    for host in hosts_list:
        save_pure_disc_helper(host)

    for pathogen in pathogens_list:
        save_pure_gen_helpler(pathogen)

    dump_trace(['<<<', 'evolve_in_population', datetime.now().isoformat()])


def chain_evolve(individuals_per_species, starting_cluster):
    # by default we will be starting with the weaker pathogens, at least for now
    dump_trace(['>>', 'chain evolve', individuals_per_species, starting_cluster,
                datetime.now().isoformat()])
    start = datetime.now()
    hosts = spawn_host_population(individuals_per_species)
    pathogens = spawn_pathogen_population(starting_cluster)
    default_budget = individuals_per_species*starting_cluster

    timer = StopWatch()
    #only base
    
    dump_evo([''])
    cross_train_iteration(hosts, pathogens, 'base', 5, timer=timer)
    
    evolve_in_population(hosts['base'], pathogens, default_budget, timer=timer)
    dump_evo([''])
    evolve_in_population(hosts['base'], pathogens, default_budget, timer=timer)
    dump_evo([''])
    evolve_in_population(hosts['base'], pathogens, default_budget, timer=timer)
    dump_evo([''])
    
    
    
    
    #cross_train_iteration(hosts, pathogens, 'base', 5, timer=timer)
    
    
    #cross_train_iteration(hosts, pathogens, 'base', 5, timer=timer)
    
    
    
    
    
    
#     dump_evo(['*********************** FIRST CROSS TRAIN AND EVOLVE IN POP RESULTS *****************************'])
#     cross_train_iteration(hosts, pathogens, 'light', 5, timer=timer)
#     evolve_in_population(hosts['light'], pathogens, default_budget, timer=timer)
#     dump_evo(['*********************** SECOND CROSS TRAIN AND EVOLVE IN POP RESULTS *****************************'])
#     cross_train_iteration(hosts, pathogens, 'PreLU', 5, timer=timer)
#     evolve_in_population(hosts['PreLU'], pathogens, default_budget, timer=timer)
#     dump_evo(['*********************** THIRD CROSS TRAIN AND EVOLVE IN POP RESULTS *****************************'])
#     cross_train_iteration(hosts, pathogens, 'base', 10, timer=timer)
#     evolve_in_population(hosts['base'], pathogens, default_budget, timer=timer)
    
    
    #EVO -- bottleneck_effect function test -- works as expected
    #hosts['light'], pathogens = bottleneck_effect(hosts['light'], pathogens)
    
      
    
        
    #cross_train_iteration(hosts, pathogens, 'light', 1, timer=timer)
    #evolve_in_population(hosts['light'], pathogens, default_budget, timer=timer)

    
    
    #cross_train_iteration(hosts, pathogens, 'PreLU', 1, timer=timer)
    #evolve_in_population(hosts['PreLU'], pathogens, default_budget, timer=timer)
    #cross_train_iteration(hosts, pathogens, 'base', 1, timer=timer)
    #evolve_in_population(hosts['base'], pathogens, default_budget, timer=timer)

    host_map = {}
    pathogen_map = {}

    for host in hosts['base']:
        host_map[host.random_tag] = [host.gen_error_map, host.current_fitness, host.real_error,
                                     host.tag_trace]

    for pathogen in pathogens:
        pathogen_map[pathogen.random_tag] = [pathogen.fitness_map, pathogen.tag_trace]

    dump_with_backup((host_map, pathogen_map), evo2_trace_dump_location)
    # pickle.dump((host_map, pathogen_map), open('evolved_2_hosts_pathogen_map.dmp', 'wb'))
    dump_trace(['<<', 'chain evolve', datetime.now().isoformat(),
                timer.get_total_time()])


def chain_evolve_with_fitness_reset(individuals_per_species, starting_cluster):
    # by default we will be starting with the weaker pathogens, at least for now
    dump_trace(['>>', 'chain evolve fit reset', individuals_per_species, starting_cluster,
                datetime.now().isoformat()])
    start = datetime.now()
    hosts = spawn_host_population(individuals_per_species)
    pathogens = spawn_pathogen_population(starting_cluster)
    default_budget = individuals_per_species*starting_cluster

    timer = StopWatch()

    cross_train_iteration(hosts, pathogens, 'light', 1, timer=timer)
    evolve_in_population(hosts['light'], pathogens, default_budget, fit_reset=True, timer=timer)
    cross_train_iteration(hosts, pathogens, 'PreLU', 1, timer=timer)
    evolve_in_population(hosts['PreLU'], pathogens, default_budget, fit_reset=True, timer=timer)
    cross_train_iteration(hosts, pathogens, 'base', 1, timer=timer)
    evolve_in_population(hosts['base'], pathogens, default_budget, fit_reset=True, timer=timer)

    host_map = {}
    pathogen_map = {}

    #TODO: sample all the images, perform a massive cross-match

    for host in hosts['base']:
        host_map[host.random_tag] = [host.gen_error_map, host.current_fitness, host.real_error,
                                     host.tag_trace]

    for pathogen in pathogens:
        pathogen_map[pathogen.random_tag] = [pathogen.fitness_map, pathogen.tag_trace]

    dump_with_backup((host_map, pathogen_map), evo2_trace_dump_location)
    # pickle.dump((host_map, pathogen_map), open('evolved_2_hosts_pathogen_map.dmp', 'wb'))
    dump_trace(['<<', 'chain evolve fit reset', datetime.now().isoformat(),
                timer.get_total_time()])


def brute_force_training(restarts, epochs):
    dump_trace(['>>', 'brute-force',
                restarts, epochs,
                datetime.now().isoformat()])

    timer = StopWatch()

    dump_trace(['>>>', 'brute-force',
                restarts, epochs,
                datetime.now().isoformat()])

    print('bruteforcing starts')
    hosts = spawn_host_population(restarts)['base']
    pathogens = spawn_pathogen_population(restarts)
    
    
    dump_test(['1/2 - INITIALLY IN BRUTE FORCE TRAINING'])
    
    for host in hosts:
        dump_test(['DISCRIMINATOR WITH RANDOM TAG: ', host.random_tag, ' WIN RATE: ', host.win_rate,\
                   ' CURRENT FITNESS: ', host.current_fitness, ' SKILL RATING TABLE: ', host.skill_rating_games,\
                   ' AND GEN ERROR MAP: ', host.gen_error_map,\
                    'TAG TRACE: ', host.tag_trace, 'UNIQUE KEY:', host.key])
    
    for pathogen in pathogens:
        dump_test(['GENERATOR INDEX WITH RANDOM TAG: ', pathogen.random_tag, ' WIN RATE: ', pathogen.win_rate,\
                   ' CURRENT FITNESS: ', pathogen.current_fitness, ' SKILL RATING TABLE: ', pathogen.skill_rating_games,\
                   ' AND GEN FITNESS MAP: ', pathogen.fitness_map,\
                    'TAG TRACE: ', pathogen.tag_trace, 'UNIQUE KEY:', pathogen.key])
    
    

    for (host_no, host), (pathogen_no, pathogen) in zip(enumerate(hosts), enumerate(pathogens)):

        timer.start()

        arena = Arena(environment=environment,
                  generator_instance=pathogen,
                  discriminator_instance=host,
                  generator_optimizer_partial=gen_opt_part,
                  discriminator_optimizer_partial=disc_opt_part)

        dump_trace(['pre-train:',
                    host_no, pathogen_no,
                    arena.discriminator_instance.random_tag,
                    arena.generator_instance.random_tag, ])

        arena.cross_train(epochs)

        timer.stop()

        arena.sample_images()

        current_fid, current_is = calc_single_fid_is(arena.generator_instance.random_tag)

        dump_trace(['sampled images from',
                    pathogen_no,
                    arena.generator_instance.random_tag,
                    current_fid, current_is, arena.generator_instance.current_fitness, arena.generator_instance.state])
        
        
        #EVOOOOOOOOOOOO
        dump_trace(['against host',
                    host_no,
                    arena.discriminator_instance.random_tag, arena.discriminator_instance.current_fitness,\
                    arena.discriminator_instance.state])
        
        
        timer.start()

        arena_match_results = arena.match()

        timer.stop()

        dump_trace(['post-cross-train and match:',
                    host_no, pathogen_no,
                    arena.discriminator_instance.random_tag,
                    arena.generator_instance.random_tag,
                    arena_match_results[0], arena_match_results[1],
                    arena.discriminator_instance.current_fitness,
                    arena.generator_instance.current_fitness])
                    

        print("%s: real_err: %s, gen_err: %s" % (
            arena.generator_instance.random_tag,
            arena_match_results[0], arena_match_results[1]))
    
    

    #EVO
    update_fitnesses(hosts)
    update_fitnesses(pathogens) 
    
    
    dump_test(['2/2 - LASTLY IN BRUTE FORCE TRAINING (AFTER UNIQUE SKILL RATING UPDATE)'])
    
    for host in hosts:
        dump_test(['DISCRIMINATOR WITH RANDOM TAG: ', host.random_tag, ' WIN RATE: ', host.win_rate,\
                   ' CURRENT FITNESS: ', host.current_fitness, ' SKILL RATING TABLE: ', host.skill_rating_games,\
                   ' AND GEN ERROR MAP: ', host.gen_error_map,\
                    'TAG TRACE: ', host.tag_trace, 'UNIQUE KEY:', host.key])
    
    for pathogen in pathogens:
        dump_test(['GENERATOR INDEX WITH RANDOM TAG: ', pathogen.random_tag, ' WIN RATE: ', pathogen.win_rate,\
                   ' CURRENT FITNESS: ', pathogen.current_fitness, ' SKILL RATING TABLE: ', pathogen.skill_rating_games,\
                   ' AND GEN FITNESS MAP: ', pathogen.fitness_map,\
                    'TAG TRACE: ', pathogen.tag_trace, 'UNIQUE KEY:', pathogen.key])
    
    
    
    for pathogen in pathogens:
        print(pathogen.random_tag, ": ", pathogen.fitness_map)

    pathogen_map = {}

    for pathogen in pathogens:
        pathogen_map[pathogen.random_tag] = [pathogen.fitness_map, pathogen.tag_trace]

    dump_with_backup(pathogen_map, brute_force_trace_dump_location)
    # pickle.dump(pathogen_map, open('brute_force_pathogen_map.dmp', 'wb'))
    dump_trace(['<<<', 'brute-force',
                datetime.now().isoformat()])
    dump_trace(['<<', 'brute-force',
                datetime.now().isoformat(),
                timer.get_total_time()])


def match_from_tags(tag_pair_set):

    dump_trace(['>>', 'matching from tags',
                datetime.now().isoformat()])

    dump_trace(['>>>', 'matching from tags',
                datetime.now().isoformat()])

    tag_pair_set.reverse()

    disc_set = set()
    gen_set = set()
    for gen_tag, disc_tag in tag_pair_set:

        try:
            disc_set.add(disc_tag)
            gen_set.add(gen_tag)
            host = Discriminator(ngpu=environment.ngpu,
                             latent_vector_size=environment.latent_vector_size,
                             discriminator_latent_maps=64,
                             number_of_colors=environment.number_of_colors).to(environment.device)

            host.resurrect(disc_tag)

            pathogen = Generator(ngpu=environment.ngpu,
                                latent_vector_size=environment.latent_vector_size,
                                generator_latent_maps=64,
                                number_of_colors=environment.number_of_colors).to(environment.device)

            pathogen.resurrect(gen_tag)

            host.to(environment.device)
            pathogen.to(environment.device)

            arena = Arena(environment=environment,
                      generator_instance=pathogen,
                      discriminator_instance=host,
                      generator_optimizer_partial=gen_opt_part,
                      discriminator_optimizer_partial=disc_opt_part)

            arena_match_results = arena.match()

            dump_trace(['post-cross-train and match:',
                        disc_tag, gen_tag,
                        arena_match_results[0], arena_match_results[1]])

        except RuntimeError:
            pass

    dump_trace(['>>>', 'matching from tags',
                datetime.now().isoformat()])

    dump_trace(['<<', 'matching from tags',
                   datetime.now().isoformat()])


if __name__ == "__main__":
    image_folder = "./image"
    image_size = 64  # TODO: test how this parameter affects training stability
    number_of_colors = 1  # TODO: remember to adjust that
    if _dataset == 'cifar10':
        number_of_colors = 3
    imtype = _dataset

    date_time = datetime.now().strftime("%d.%m.%Y-%H.%M.%S")
    trace_dump_file = 'run_trace_' + date_time + '_' + imtype + '.csv'
    
    test_dump_file = 'test_' + date_time + '.csv'



    #TODO: here is where we are going for fashion-mnist; CIFAR-10 and CelebA datasets

    # # raw size: 218x178 @ 3c
    # celeb_a_dataset = dset.CelebA(root=image_folder, download=True,
    #                                 transform=transforms.Compose([
    #                                    transforms.Resize(image_size),
    #                                    transforms.ToTensor(),
    #                                    transforms.Normalize((0.5,), (0.5,)),]))
    # raw size: 28x28 @ 1c
    fashion_mnist_dataset = dset.FashionMNIST(root=image_folder, download=True,
                                              transform=transforms.Compose([
                                           transforms.Resize(image_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5,), (0.5,)),]))

    # # raw size: 32x32 @ 3c
    cifar10_dataset = dset.CIFAR10(root=image_folder, download=True,
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,)),]))

    # raw size: 28x28 @ 1c
    mnist_dataset = dset.MNIST(root=image_folder, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,)),]))

    if _dataset == 'fashion_mnist':
        current_dataset = fashion_mnist_dataset
    elif _dataset == 'mnist':
        current_dataset = mnist_dataset
    elif _dataset == 'cifar10':
        current_dataset = cifar10_dataset
    else:
        raise Exception('unrecognized dataset type: %s' % _dataset)

    environment = GANEnvironment(current_dataset,
                                 device=cuda_device,
                                 number_of_colors=number_of_colors)

    learning_rate = 0.0002
    beta1 = 0.5

    gen_opt_part = lambda x: optim.Adam(x, lr=learning_rate, betas=(beta1, 0.999))
    disc_opt_part = lambda x: optim.Adam(x, lr=learning_rate, betas=(beta1, 0.999))

    dump_trace(['>', 'run started', datetime.now().isoformat()])

    run_start = datetime.now()

    try:
        
#         ######################################################################################
#         ######################### FIRST EXPERIMENTS -- CHAIN EVOLVE 5,5 ######################
        
#         dump_test(['FIRST CHAIN EVOLVE STARTED'])
#         dump_evo(['************** FIRST CHAIN EVOLVE ADAPTATION RESULTS ************************'])
#         chain_evolve(5, 5)
#         dump_evo([''])
#         dump_test([''])
        
#         dump_test(['SECOND CHAIN EVOLVE STARTED'])
#         dump_evo(['************** SECOND CHAIN EVOLVE ADAPTATION RESULTS ***********************'])
#         chain_evolve(5, 5)
#         dump_evo([''])
#         dump_test([''])
        
#         dump_test(['THIRD CHAIN EVOLVE STARTED'])
#         dump_evo(['************** THIRD CHAIN EVOLVE ADAPTATION RESULTS ************************'])
#         chain_evolve(5, 5)
#         dump_evo([''])
#         dump_test([''])
        
#         dump_test(['FOURTH CHAIN EVOLVE STARTED'])
#         dump_evo(['************** FOURTH CHAIN EVOLVE ADAPTATION RESULTS ***********************'])
#         chain_evolve(5, 5)
#         dump_evo([''])
#         dump_test([''])
        
#         dump_test(['FIFTH CHAIN EVOLVE STARTED'])
#         dump_evo(['************** FIFTH CHAIN EVOLVE ADAPTATION RESULTS ************************'])
#         chain_evolve(5, 5)
#         dump_test([''])
#         dump_test(['ALL CHAIN EVOLVE EXPERIMENTS COMPLETED'])
#         dump_test([''])
        
#         #######################################################################################     
        
        
#         #######################################################################################
#         ########################## SECOND EXPERIMENTS -- STOCHASTIC ROUND ROBIN 5,5 ###########
        
#         dump_test(['FIRST ROUND ROBIN RANDOMIZED STARTED'])
#         round_robin_randomized(5, 5)
#         dump_test([''])
        
#         dump_test(['SECOND ROUND ROBIN RANDOMIZED STARTED'])
#         round_robin_randomized(5, 5)
#         dump_test([''])
        
#         dump_test(['THIRD ROUND ROBIN RANDOMIZED STARTED'])
#         round_robin_randomized(5, 5)
#         dump_test([''])
        
#         dump_test(['FOURTH ROUND ROBIN RANDOMIZED STARTED'])
#         round_robin_randomized(5, 5)
#         dump_test([''])
        
#         dump_test(['FIFTH ROUND ROBIN RANDOMIZED STARTED'])
#         round_robin_randomized(5, 5)
#         dump_test([''])
#         dump_test(['ALL ROUND ROBIN RANDOMIZED EXPERIMENTS COMPLETED'])
#         dump_test([''])
        
#         #######################################################################################
               
        
#         #######################################################################################
#         ########################## THIRD EXPERIMENTS -- DETERMINISTIC ROUND ROBIN 5,5 #########
        
#         dump_test(['FIRST ROUND ROBIN DETERMINISTIC STARTED'])
#         round_robin_deterministic(5, 5)
#         dump_test([''])
        
#         dump_test(['SECOND ROUND ROBIN DETERMINISTIC STARTED'])
#         round_robin_deterministic(5, 5)
#         dump_test([''])
        
#         dump_test(['THIRD ROUND ROBIN DETERMINISTIC STARTED'])
#         round_robin_deterministic(5, 5)
#         dump_test([''])
        
#         dump_test(['FOURTH ROUND ROBIN DETERMINISTIC STARTED'])
#         round_robin_deterministic(5, 5)
#         dump_test([''])
        
#         dump_test(['FIFTH ROUND ROBIN DETERMINISTIC STARTED'])
#         round_robin_deterministic(5, 5)
#         dump_test([''])
#         dump_test(['ALL ROUND ROBIN DETERMINISTIC EXPERIMENTS COMPLETED'])
#         dump_test([''])
        
#         #######################################################################################
        
        
#         #######################################################################################
#         ########################## FOURTH EXPERIMENTS -- BRUTE FORCE TRAINING 10,15 ###########
        
#         dump_test(['FIRST BRUTE FORCE STARTED'])
#         brute_force_training(10, 15)
#         dump_test([''])
        
#         dump_test(['SECOND BRUTE FORCE STARTED'])
#         brute_force_training(10, 15)
#         dump_test([''])
        
#         dump_test(['THIRD BRUTE FORCE STARTED'])
#         brute_force_training(10, 15)
#         dump_test([''])
        
#         dump_test(['FOURTH BRUTE FORCE STARTED'])
#         brute_force_training(10, 15)
#         dump_test([''])
        
#         dump_test(['FIFTH BRUTE FORCE STARTED'])
#         brute_force_training(10, 15)
#         dump_test([''])
#         dump_test(['ALL ROUND BRUTE FORCE EXPERIMENTS COMPLETED'])
        
#         ######################################################################################
        

        
        
        pass

    except Exception as exc:
        try:
            print('ERROR IN MAIN LOOP, PUT LOGGING CODE BACK FOR DEBUG [Amir]')
                      
#             dump_test([logger.exception("1. EXCEPTION IN MAIN")])
#             dump_test([logger.debug("2. DEBUG IN MAIN")])
#             dump_test([logger.info("3. INFO IN MAIN")])
            
#             logger.exception("1. EXCEPTION IN MAIN")
#             logger.debug("2. DEBUG IN MAIN")
#             logger.info("3. INFO IN MAIN")
            
            
#             logger.error(exc, exc_info=True)
        except (AttributeError, smtplib.SMTPServerDisconnected):
            smtp_error_bail_out()
        raise

    dump_trace(['<', 'run completed', datetime.now().isoformat()])

    successfully_completed(start_date=run_start, start_device=cuda_device)

    # gen = Generator(ngpu=environment.ngpu,
    #                 latent_vector_size=environment.latent_vector_size,
    #                 generator_latent_maps=64,
    #                 number_of_colors=environment.number_of_colors).to(environment.device)
    #
    # disc = Discriminator(ngpu=environment.ngpu,
    #                      latent_vector_size=environment.latent_vector_size,
    #                      discriminator_latent_maps=64,
    #                      number_of_colors=environment.number_of_colors).to(environment.device)
    #
    #
    # arena = Arena(environment=environment,
    #               generator_instance=gen,
    #               discriminator_instance=disc,
    #               generator_optimizer_partial=gen_opt_part,
    #               discriminator_optimizer_partial=disc_opt_part,
    #               )
    #
    # arena.cross_train(5)
    # arena.sample_images()
    # print(arena.generator_instance.random_tag)
    # print(arena.match())
