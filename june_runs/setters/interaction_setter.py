from june.interaction import Interaction
from june.paths import configs_path

default_interaction_config = configs_path / "defaults/interaction/interaction.yaml"


class InteractionSetter:
    def __init__(
        self,
        alpha_physical=None,
        betas=None,
        susceptibilities_by_age=None,
        population=None,
        baseline_interaction_path=None,
    ):
        self.alpha_physical = alpha_physical
        self.betas = betas
        self.susceptibilities_by_age = susceptibilities_by_age
        self.population = population
        self.baseline_interaction_path = baseline_interaction_path or default_interaction_config

    @classmethod
    def from_parameters(
        cls, parameters: dict, baseline_interaction_path: str, population
    ):
        interaction_parameters = parameters["interaction"]
        alpha_physical = interaction_parameters.get("alpha_physical", None)
        betas = interaction_parameters.get("betas", None)
        alpha_physical = interaction_parameters.get("alpha_physical", None)
        susceptibilities = interaction_parameters.get("susceptibilities", None)
        return cls(
            betas=betas,
            alpha_physical=alpha_physical,
            susceptibilities_by_age=susceptibilities,
            baseline_interaction_path=baseline_interaction_path,
            population=population,
        )

    def make_interaction(self):
        interaction = Interaction.from_file(
            config_filename=self.baseline_interaction_path, population=self.population
        )
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
