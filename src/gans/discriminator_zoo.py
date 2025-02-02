import torch
import torch.nn as nn
from src.gans.nn_structure import NetworkStructure
from random import sample
import string
import pickle
import sys
from src.mongo_interface import pure_disc_from_random_tag
import io
from configs import cuda_device

from src.glicko2 import glicko2

#from src.evo_helpers import map_transform

import uuid


#Discriminators implementations

#Collection of several implementation of discriminator, sll designed for images as input
#Storage-ressuraction boilerplate

char_set = string.ascii_uppercase + string.digits


# TODO: make sure that the saving and resurrection are done to CPU at first and then sent to CUDAs
torch.cuda.set_device(cuda_device)

def generate_hyperparameter_key(_self):
    key = {'random_tag': _self.random_tag,
           'disc_type': type(_self).__name__,
           'disc_latent_params': _self.discriminator_latent_maps}
    return key


def storage_representation(_self):
    _self.to(torch.device('cpu'))
    key = _self.generate_hyperparameter_key()
    payload = {'encounter_trace': _self.encounter_trace,
                'disc_state': pickle.dumps(_self.state_dict()),
                'self_error': _self.real_error,
                'gen_error_map': _self.gen_error_map,
                # TODO: Map needs to include both the generator errors and virulence factors.
                'current_fitness': _self.current_fitness}

    key.update(payload)
    _self.to(torch.device(cuda_device))

    return key



def resurrect(_self, random_tag):
    _self.to(torch.device('cpu'))
    stored_disc = pure_disc_from_random_tag(random_tag)

    # print(sys.getsizeof(stored_disc))

    if stored_disc['disc_type'] != type(_self).__name__:
        raise Exception('Wrong class: expected %s, got %s' % (type(_self).__name__,
                                                              stored_disc['disc_type']))
    _self.random_tag = random_tag
    _self.generator_latent_maps = stored_disc['disc_latent_params']
    _self.encounter_trace = stored_disc['encounter_trace']
    # print(torch.cuda.current_device())

    # print(sys.getsizeof(stored_disc['disc_state']) / 1024. / 1024.)
    # with torch.device('cpu'):
    _self.load_state_dict(pickle.loads(stored_disc['disc_state']))
    # fake_file = io.BytesIO(stored_disc['disc_state'])
    # _self.load_state_dict(torch.load(fake_file, map_location=torch.device('cpu')))
    # print('encounter_trace:', _self.encounter_trace)
    _self.real_error = stored_disc['self_error']
    _self.gen_error_map = stored_disc['gen_error_map']
    _self.current_fitness = stored_disc['current_fitness']
    _self.to(torch.device(cuda_device))


class GaussianNoise(nn.Module):

    def __init__(self, sigma=0.1, device=cuda_device):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = True
        self.noise = torch.tensor(0).to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Discriminator(nn.Module):

    def __init__(self, ngpu, latent_vector_size, discriminator_latent_maps, number_of_colors,
                 autoimmunity=20):
        super(Discriminator, self).__init__()
        self.tag = "disc_base"
        self.random_tag = ''.join(sample(char_set * 10, 10))
        self.ngpu = ngpu
        self.latent_vector_size = latent_vector_size
        self.discriminator_latent_maps = discriminator_latent_maps
        self.number_of_colors = number_of_colors
        self.real_error = 1.
        self.gen_error_map = {}
        #self.current_fitness = 0.
        self.encounter_trace = []  # ((type, id, training_trace, match score))
        self.tag_trace = [self.random_tag]
        self.autoimmunity = autoimmunity
        # TODO: Gaussian noise injection
        # self.noise = GaussianNoise()
                
        self.win_rate = 0
        self.glicko = glicko2.Glicko2(mu=1500, phi=350, sigma=0.06, tau=0.3)
        self.skill_rating = self.glicko.create_rating()
        self.skill_rating_games = []
        
        self.current_fitness = self.skill_rating.mu
        
        self.silent_map = False
        self.parent_silent_map = False
        self.coadaptation = False
        self.parent_coadaptation = False
        
        self.key = uuid.uuid4().hex
        
        self.state = 'No adaptation'
        
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(in_channels=self.number_of_colors,
                      out_channels=self.discriminator_latent_maps,
                      kernel_size=4,
                      stride=2,  # affects the size of the out map (divides)
                      padding=1,  # affects the size of the out map
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.discriminator_latent_maps,
                      self.discriminator_latent_maps * 2,
                      4, 2, 1,
                      bias=False),
            nn.BatchNorm2d(self.discriminator_latent_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.discriminator_latent_maps * 2,
                      self.discriminator_latent_maps * 4,
                      4, 2, 1,
                      bias=False),
            nn.BatchNorm2d(self.discriminator_latent_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.discriminator_latent_maps * 4,
                      self.discriminator_latent_maps * 8,
                      4, 2, 1,
                      bias=False),
            nn.BatchNorm2d(self.discriminator_latent_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.discriminator_latent_maps * 8,
                      out_channels=1,
                      kernel_size=4,
                      stride=1,
                      padding=0,
                      bias=False),

            nn.Sigmoid()
        )

    def bind_nn_structure(self, network: NetworkStructure):
        #TODO: check if in/out dimensions are consistent
        self.main = nn.Sequential(network.compile())

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            # input = self.noise(input)
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            # input = self.noise(input)
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

    def generate_hyperparameter_key(self):
        return generate_hyperparameter_key(self)

    def save_instance_state(self):
        return storage_representation(self)

    def size_on_disc(self):
        return count_parameters(self.main)

    #EVO
    def bump_random_tag(self):
        temp_a = self.silent_map #save adaptation of parent-to-be
        temp_c = self.coadaptation #save whether the parent-to-be was silently adapted
        
        self.random_tag = ''.join(sample(char_set * 10, 10))
        self.tag_trace += [self.random_tag]
        
        self.silent_map = False
        self.parent_silent_map = temp_a
        self.coadaptation = False
        self.parent_coadaptation = temp_c
        self.gen_error_map.clear()

    def resurrect(self, random_tag):
        resurrect(self, random_tag)

        
    def calc_win_rate(self, disc_decision_on_real, disc_decision_on_fake):
        self.win_rate += (sum((disc_decision_on_real > 0.5).float()) + \
                          sum((disc_decision_on_fake < 0.5).float())).item() /\
        (len(disc_decision_on_real) + len(disc_decision_on_fake))
    
    
    #creates a rating object for its adversarial (its opponent in a specific game)
    #appends (self.win_rate, adv.skill_rating) to its skill_rating_games []
    def calc_skill_rating(self, adversarial):
        rating = self.glicko.create_rating(mu=adversarial.skill_rating.mu,\
                                           phi=adversarial.skill_rating.phi, sigma=adversarial.skill_rating.sigma)
        
        self.skill_rating_games.append((self.win_rate, rating))

    
    #assigns a skill_rating to self, and resets own skill_rating_games table (got the skill rating from all those stored games played)
    def finish_calc_skill_rating(self):
        self.skill_rating = self.glicko.rate(self.skill_rating, self.skill_rating_games)
        self.skill_rating_games = []



class Discriminator_light(nn.Module):

    def __init__(self, ngpu, latent_vector_size, discriminator_latent_maps, number_of_colors,
                 autoimmunity=20):
        super(Discriminator_light, self).__init__()
        self.tag = 'disc_light'
        self.random_tag = ''.join(sample(char_set * 10, 10))
        self.ngpu = ngpu
        self.latent_vector_size = latent_vector_size
        self.discriminator_latent_maps = discriminator_latent_maps
        self.number_of_colors = number_of_colors
        self.real_error = 1.
        self.gen_error_map = {}
        #self.current_fitness = 0.
        self.encounter_trace = []  # ((type, id, training_trace, match score))
        self.tag_trace = [self.random_tag]
        self.autoimmunity = autoimmunity
        # TODO: Gaussian noise injection
        # self.noise = GaussianNoise()
                
        self.win_rate = 0
        self.glicko = glicko2.Glicko2(mu=1500, phi=350, sigma=0.06, tau=0.3)
        self.skill_rating = self.glicko.create_rating()
        self.skill_rating_games = []
        
        self.current_fitness = self.skill_rating.mu
        
        self.silent_map = False
        self.parent_silent_map = False
        self.coadaptation = False
        self.parent_coadaptation = False
        
        self.key = uuid.uuid4().hex
        
        self.state = 'No adaptation'
        
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(in_channels=self.number_of_colors,
                      out_channels=self.discriminator_latent_maps,
                      kernel_size=4,
                      stride=2,  # affects the size of the out map (divides)
                      padding=1,  # affects the size of the out map
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.discriminator_latent_maps, self.discriminator_latent_maps * 2, 4, 2, 1,
                      bias=False),
            nn.BatchNorm2d(self.discriminator_latent_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Linear((self.discriminator_latent_maps*2) * 16 * 16,
                      (self.discriminator_latent_maps*2) * 16 * 16),
            nn.BatchNorm2d(self.discriminator_latent_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.discriminator_latent_maps * 2, self.discriminator_latent_maps * 4, 4, 2,
                      1,
                      bias=False),
            nn.BatchNorm2d(self.discriminator_latent_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.discriminator_latent_maps * 4, self.discriminator_latent_maps * 8, 4, 2,
                      1, bias=False),
            nn.BatchNorm2d(self.discriminator_latent_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.discriminator_latent_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def bind_nn_structure(self, network: NetworkStructure):
        #TODO: check if in/out dimensions are consistent
        self.main = nn.Sequential(network.compile())

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            # input = self.noise(input)
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            # input = self.noise(input)
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

    def generate_hyperparameter_key(self):
        return generate_hyperparameter_key(self)

    def save_instance_state(self):
        return storage_representation(self)

    def size_on_disc(self):
        return count_parameters(self.main)

    #EVO
    def bump_random_tag(self):
        temp_a = self.silent_map #save adaptation of parent-to-be
        temp_c = self.coadaptation #save whether the parent-to-be was silently adapted
        
        self.random_tag = ''.join(sample(char_set * 10, 10))
        self.tag_trace += [self.random_tag]
        
        self.silent_map = False
        self.parent_silent_map = temp_a
        self.coadaptation = False
        self.parent_coadaptation = temp_c
        self.gen_error_map.clear()

    def resurrect(self, random_tag):
        resurrect(self, random_tag)


    def calc_win_rate(self, disc_decision_on_real, disc_decision_on_fake):
        self.win_rate += (sum((disc_decision_on_real > 0.5).float()) + \
                          sum((disc_decision_on_fake < 0.5).float())).item() /\
        (len(disc_decision_on_real) + len(disc_decision_on_fake))
    
    
    #creates a rating object for its adversarial (its opponent in a specific game)
    #appends (self.win_rate, adv.skill_rating) to its skill_rating_games []
    def calc_skill_rating(self, adversarial):
        rating = self.glicko.create_rating(mu=adversarial.skill_rating.mu,\
                                           phi=adversarial.skill_rating.phi, sigma=adversarial.skill_rating.sigma)
        
        self.skill_rating_games.append((self.win_rate, rating))
    
    #assigns a skill_rating to self, and resets own skill_rating_games table (got the skill rating from all those stored games played)
    def finish_calc_skill_rating(self):
        self.skill_rating = self.glicko.rate(self.skill_rating, self.skill_rating_games)
        self.skill_rating_games = []
        


class Discriminator_PReLU(nn.Module):

    def __init__(self, ngpu, latent_vector_size, discriminator_latent_maps, number_of_colors,
                 autoimmunity=20):
        super(Discriminator_PReLU, self).__init__()
        self.tag = "disc_PReLU"
        self.random_tag = ''.join(sample(char_set * 10, 10))
        self.ngpu = ngpu
        self.latent_vector_size = latent_vector_size
        self.discriminator_latent_maps = discriminator_latent_maps
        self.number_of_colors = number_of_colors
        self.real_error = 1.
        self.gen_error_map = {}
        #self.current_fitness = 0.
        self.encounter_trace = []  # ((type, id, training_trace, match score))
        self.tag_trace = [self.random_tag]
        self.autoimmunity = autoimmunity
        # TODO: Gaussian noise injection
        # self.noise = GaussianNoise()
        
        self.win_rate = 0
        self.glicko = glicko2.Glicko2(mu=1500, phi=350, sigma=0.06, tau=0.3)
        self.skill_rating = self.glicko.create_rating()
        self.skill_rating_games = []
        
        self.current_fitness = self.skill_rating.mu
        
        self.silent_map = False
        self.parent_silent_map = False
        self.coadaptation = False
        self.parent_coadaptation = False
        
        self.key = uuid.uuid4().hex
        
        self.state = 'No adaptation'
        
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(in_channels=self.number_of_colors,
                      out_channels=self.discriminator_latent_maps,
                      kernel_size=4,
                      stride=2,  # affects the size of the out map (divides)
                      padding=1,  # affects the size of the out map
                      bias=False),
            nn.PReLU(),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.discriminator_latent_maps, self.discriminator_latent_maps * 2, 4, 2,
                      1,
                      bias=False),
            nn.BatchNorm2d(self.discriminator_latent_maps * 2),
            nn.PReLU(),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.discriminator_latent_maps * 2, self.discriminator_latent_maps * 4, 4,
                      2,
                      1,
                      bias=False),
            nn.BatchNorm2d(self.discriminator_latent_maps * 4),
            nn.PReLU(),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.discriminator_latent_maps * 4, self.discriminator_latent_maps * 8, 4,
                      2,
                      1, bias=False),
            nn.BatchNorm2d(self.discriminator_latent_maps * 8),
            nn.PReLU(),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.discriminator_latent_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def bind_nn_structure(self, network: NetworkStructure):
        # TODO: check if in/out dimensions are consistent
        self.main = nn.Sequential(network.compile())

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            # input = self.noise(input)
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            # input = self.noise(input)
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

    def generate_hyperparameter_key(self):
        return generate_hyperparameter_key(self)

    def save_instance_state(self):
        return storage_representation(self)

    def size_on_disc(self):
        return count_parameters(self.main)

    #EVO
    def bump_random_tag(self):
        temp_a = self.silent_map #save adaptation of parent-to-be
        temp_c = self.coadaptation #save whether the parent-to-be was silently adapted
        
        self.random_tag = ''.join(sample(char_set * 10, 10))
        self.tag_trace += [self.random_tag]
        
        self.silent_map = False
        self.parent_silent_map = temp_a
        self.coadaptation = False
        self.parent_coadaptation = temp_c
        self.gen_error_map.clear()

    def resurrect(self, random_tag):
        resurrect(self, random_tag)

        
    def calc_win_rate(self, disc_decision_on_real, disc_decision_on_fake):
        self.win_rate += (sum((disc_decision_on_real > 0.5).float()) + \
                          sum((disc_decision_on_fake < 0.5).float())).item() /\
        (len(disc_decision_on_real) + len(disc_decision_on_fake))
    
    
    #creates a rating object for its adversarial (its opponent in a specific game)
    #appends (self.win_rate, adv.skill_rating) to its skill_rating_games []
    def calc_skill_rating(self, adversarial):
        rating = self.glicko.create_rating(mu=adversarial.skill_rating.mu,\
                                           phi=adversarial.skill_rating.phi, sigma=adversarial.skill_rating.sigma)
        
        self.skill_rating_games.append((self.win_rate, rating))
        
    #assigns a skill_rating to self, and resets own skill_rating_games table (got the skill rating from all those stored games played)
    def finish_calc_skill_rating(self):
        self.skill_rating = self.glicko.rate(self.skill_rating, self.skill_rating_games)
        self.skill_rating_games = []
        