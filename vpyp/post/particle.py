from tools import sample_multinomial

from copy import deepcopy
import numpy as np


class Particle:
    def __init__(self, tag_model, word_model, sentence, n_tags):
        self.tag_model = deepcopy(tag_model)
        self.word_model = deepcopy(word_model)

        self.tag_assignments = []
        self.sentence = sentence

        self.state = 0
        self.weight = 1

        self.n_words = len(sentence)
        self.n_tags = n_tags

    def proposal_prob(self, tag):
        return self.tag_model.prob(ctx=self.context(), w=tag) * self.word_model.prob(ctx=self.context(), w=tag)

    def proposal_distribution(self):
        proposal_distribution = [self.proposal_prob(t) for t in range(self.n_tags)]
        return [proposal_prob / sum(proposal_distribution)
                for proposal_prob in range(len(proposal_distribution))]

    def pick_tag(self, proposal_distribution):
        return sample_multinomial(n=1, dist=proposal_distribution)

    def update_tables(self, tag):
        self.tag_model.decrement(ctx=self.context(), w=tag)
        self.word_model.decrement(ctx=tag, w=self.word())

    def context(self):
        return self.tag_assignments[max(self.state - 1, self.n_words):min((self.state + 1), self.n_words)]

    def word(self):
        return self.sentence[self.state][1]

    def assign_tag(self, tag):
        self.update_tables(tag)
        self.tag_assignments.append(tag)

    def compute_weight(self, proposal_prob, tag):
        return self.tag_model.prob(ctx=self.context(), w=tag) / proposal_prob

    def particle_filter_step(self):

        # Store the old proposals for the not yet updated model
        proposal_dist = self.proposal_distribution()

        # Pick a new tag according to the proposal and assign it
        tag = self.pick_tag(proposal_dist)
        self.assign_tag(tag)

        # Update the particle weight
        self.weight *= self.compute_weight(proposal_prob=proposal_dist[tag], tag=tag)

    def particle_filter(self):
        for _ in range(self.n_words) :
            self.particle_filter_step()


class Particles:
    def __init__(self, n_particles, tag_model, word_model, sentence, n_tags):
        self.particles = [Particle(tag_model, word_model, sentence, n_tags) for _ in range(n_particles)]
        self.weights = np.ones(self.n_particles)

        self.n_particles = n_particles

    def sample_particle(self):
        # TODO : Check the particle sampling
        weights = [particle.weight for particle in self.particles]
        weights = [weight / sum(weights) for weight in weights]
        sampled_idx = sample_multinomial(n=1, dist=weights)
        return self.particles[sampled_idx]

    def particle_filter(self):
        """" Returns the tag assignments sampled by the particle filter"""
        for particle in self.particles:
            particle.particle_filter()
        return self.sample_particle().tag_assignments
