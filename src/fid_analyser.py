import os
import sys
from random import shuffle
from itertools import combinations
from src.fid_calc.fid_score import calculate_fid_given_paths  # TRACING: usage of fid > fid_is
from src.fid_calc.fid_score import calculate_fid_and_is_given_paths
from src.fid_is_calc.both import get_inception_score_and_fid
import pickle
import datetime
from configs import fid_samples_location

balancing_folders_location = fid_samples_location
# fid_command = '/home/kucharav/Documents/pytorch-fid-master/fid_score.py'
after_datetime = datetime.datetime.now() - datetime.timedelta(days=1)



#import logging

#logging.basicConfig(filename="test.csv", level=logging.DEBUG)

exception_dump_file = "Exception_dump.csv"

def dump_for_exception(payload_list):
    if not os.path.isfile(exception_dump_file):
        open(exception_dump_file, 'w')

    with open(exception_dump_file, 'a') as destination:
        writer = csv.writer(destination, delimiter='\t')
        writer.writerow(payload_list)



        
        
def calc_single_fid_is(random_tag):
    total_path = os.path.join(balancing_folders_location, random_tag)

    if os.path.isdir(total_path):
        current_real = balancing_folders_location + '/' + random_tag + '/' + 'real'
        current_fake = balancing_folders_location + '/' + random_tag + '/' + 'fake'

        try:
            '''
            #This may be omitted, replaced by what is below
            fid_value = calculate_fid_given_paths([current_real, current_fake],
                                                      batch_size=64,
                                                      cuda=True,
                                                      dims=2048)
                        
            print('fid compute', random_tag, ': ', fid_value)
            '''
            
            fid_val, is_val = calculate_fid_and_is_given_paths([current_real, current_fake],
                                                      batch_size=64,
                                                      cuda=True,
                                                      dims=2048)
            
            #print('new fid and is compute', random_tag, ': ', (fid_val, is_val))
            
            return fid_val, is_val

        except Exception as e:
            '''                        
            dump_for_exception([logging.error("Exception occured in calc_single_fid_is (1. logging.error)", exc_info=True), \
                                logging.debug("Exception occured in calc_single_fid_is (2. logging.debug)", exc_info=True)])
            
            logging.error("Exception occured in calc_single_fid_is (1. logging.error)", exc_info=True)
            logging.debug("Exception occured in calc_single_fid_is (2. logging.debug)", exc_info=True)
            
            
            logger = logging.getLogger()
            
            dump_for_exception([logger.exception("Exception (3. logger.exception)"), logger.debug("Exception (4. logger.debug)"), \
                                logger.info("Exception (5. logger.info)")])            
                      
            logger.exception("Exception (3. logger.exception)")
            logger.debug("Exception (4. logger.debug)")
            logger.info("Exception (5. logger.info)")
            '''
            
            print("Unexpected error:", sys.exc_info())

    return -1, -1

#Changed code to create, fill and return inception score map : is_map, returns 3 values now
def calc_gen_fids_is():
    random_tag_list = []
    fid_map = {}
    is_map = {}
    
    for random_tag in os.listdir(balancing_folders_location):
        if os.path.getmtime(random_tag) > after_datetime:
            random_tag_list.append(random_tag)

            current_real = balancing_folders_location + '/' + random_tag + '/' + 'real'
            current_fake = balancing_folders_location + '/' + random_tag + '/' + 'fake'

            try:
                fid_value, is_value = calculate_fid_and_is_given_paths([current_real, current_fake],
                                                              batch_size=64,
                                                              cuda=True,
                                                              dims=2048)

                fid_map[random_tag] = fid_value
                is_map[random_tag]  = is_value
                print(random_tag, ': 1.FID: ', fid_value, '\n 2.IS: ', is_value)
                
            except:
                print("Unexpected error:", sys.exc_info())

    return is_map, fid_map, random_tag_list


def calc_reals_fid(random_tag_list):
    real_comparison = []
    shuffle(random_tag_list)
    blocker = 20
    for i, (random_tag_1, random_tag_2) in enumerate(combinations(random_tag_list, 2)):
        if i > blocker:
            break
        current_1 = balancing_folders_location + '/' + random_tag_1 + '/' + 'real'
        current_2 = balancing_folders_location + '/' + random_tag_2 + '/' + 'real'

        try:
            fid_value = calculate_fid_given_paths([current_1, current_2],
                                                      batch_size=64,
                                                      cuda=True,
                                                      dims=2048)
            real_comparison.append(fid_value)
            print('real to real sample', ': ', fid_value)
        except:
            print("Unexpected error:", sys.exc_info()[0])

    return real_comparison


if __name__ == "__main__":
    
    is_map, fid_map, random_tag_list = calc_gen_fids_is() # added code to get the inception score map
    real_comparison = calc_reals_fid(random_tag_list)

    
    if os.path.isfile('fid_scores.dmp'):
        old_fid_map, old_real_comparison = pickle.load((fid_map, real_comparison), open('fid_scores.dmp', 'rb'))
        fid_map.update(old_fid_map)
        real_comparison += old_real_comparison

    pickle.dump((fid_map, real_comparison), open('fid_scores.dmp', 'wb'))
    
    
    # Added code to dump inception scores!  (do we need real_comparaison ?)
    if os.path.isfile('inception_scores.dmp'):
        old_is_map = pickle.load((is_map), open('inception_scores.dmp', 'rb'))
        is_map.update(old_is_map)

    pickle.dump((is_map), open('inception_scores.dmp', 'wb'))
    
    