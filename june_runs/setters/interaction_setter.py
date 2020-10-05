from june.interaction import Interaction


class InteractionSetter:
    def __init__(
        self,
        alpha_physical=None,
        betas=None,
        susceptibilities_by_age=None,
        population=None,
    ):
        self.alpha_physical = alpha_physical
        self.betas = betas
        self.susceptibilities_by_age = susceptibilities_by_age
        self.population = population

    def make_interaction(self):
        interaction = Interaction.from_file(population=self.population)
        if self.alpha_physical is not None:
            interaction.alpha_physical = self.alpha_physical
        if self.betas is not None:
            allowed_betas = interaction.beta
            for key, value in self.betas.items():
                if key not in allowed_betas:
                    raise ValueError(
                        f"Trying to change a beta for a non-existing group."
                    )
                else:
                    interaction.beta[key] = value
        # susceptibility
        if self.susceptibilities_by_age is not None:
            interaction.set_population_susceptibilities(
                population=self.population,
                susceptibilities_by_age=self.susceptibilities_by_age,
            )
        return interaction
