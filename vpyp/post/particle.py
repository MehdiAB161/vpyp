from tools import sample_multinomial
from corpus import ngrams

# TODO : add particle 0


class Particle:
    def __init__(self, tag_model, word_model, sentence, n_tags, model_type):
        """"
        model_type : "word" or "sentence"
        """
        # self.tag_model = deepcopy(tag_model)
        # self.word_model = deepcopy(word_model)

        self.tag_model = tag_model
        self.word_model = word_model

        self.tag_assignments = []
        self.sentence = sentence
        self.words = [pair[1] for pair in sentence]

        self.state = 0
        self.weight = 1

        self.n_words = len(sentence)
        self.n_tags = n_tags
        self.model_type = model_type

    # TODO : Check if there is an underflow
    def proposal_prob(self, tag):
        if self.model_type == "sentence":
            return self.tag_model.prob(ctx=self.context(), w=tag) * self.word_model.prob(ctx=tag, w=self.word())
        if self.model_type == "word":
            pass
        # TODO : add word proposal wor the word type model

    def proposal_distribution(self):
        proposal_distribution = [self.proposal_prob(t) for t in range(self.n_tags)]
        proposal_distribution = [proposal_prob / sum(proposal_distribution)
                for proposal_prob in proposal_distribution]
        return proposal_distribution

    @staticmethod
    def pick_tag(proposal_distribution):
        return sample_multinomial(n=1, dist=proposal_distribution)[0]

    def increment_tables(self, tag):
        # TODO : Check the format uniformity for word tags
        # TODO : Check the model tags
        # TODO : Increment ?
        self.tag_model.increment(ctx=self.context(), w=tag)
        self.word_model.increment(ctx=tag, w=self.word())

    def context(self):
        return list(ngrams(self.tag_assignments, 3))[self.state][:-1]

    def word(self):
        return self.words[self.state]

    def assign_tag(self, tag):
        # Update the seating arrangements for the word type model only
        if self.model_type == "word":
            self.increment_tables(tag)
        self.tag_assignments.append(tag)

    def compute_weight(self, proposal_prob, tag):
        return self.tag_model.prob(ctx=self.context(), w=tag) / proposal_prob

    def particle_filter_step(self):

        # Store the old proposals for the not yet updated model
        proposal_dist = self.proposal_distribution()

        # Pick a new tag according to the proposal and assign it
        tag = self.pick_tag(proposal_dist)

        # Update the tags
        self.assign_tag(tag)

        # Update the particle weight
        self.weight *= self.compute_weight(proposal_prob=proposal_dist[tag], tag=tag)

        self.update_state()

    def particle_filter(self):
        for _ in range(self.n_words) :
            self.particle_filter_step()

    def update_state(self):
        self.state += 1


class Particles:
    def __init__(self, n_particles, tag_models, word_models, sentence, n_tags, model_type):

        if model_type == "sentence":
            # The seats are updated until all the sequence is picked, only a model is necessary for all the particles
            self.particles = [Particle(tag_models, word_models, sentence, n_tags, model_type)
                              for _ in range(n_particles)]
        if model_type == "word":
            self.particles = [Particle(tag_models[i], word_models[i], sentence, n_tags, model_type)
                              for i in range(n_particles)]
        # self.particles = Parallel(n_jobs=4)(delayed(Particle)(tag_model, word_model, sentence, n_tags)
        #                                     for _ in range(n_particles))
        self.n_particles = n_particles

    def sample_particle(self):
        # TODO : Check the particle sampling
        # TODO : Check the weights
        weights = [particle.weight for particle in self.particles]
        weights = [weight / sum(weights) for weight in weights]
        sampled_idx = sample_multinomial(n=1, dist=weights)[0]
        return self.particles[sampled_idx]

    def particle_filter(self):
        """" Returns the tag assignments sampled by the particle filter"""
        for particle in self.particles:
            particle.particle_filter()
        filtered_particle = self.sample_particle()
        return filtered_particle
