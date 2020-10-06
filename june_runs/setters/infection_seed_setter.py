from june.infection_seed import InfectionSeed, Observed2Cases


class InfectionSeedSetter:
    def __init__(
        self,
        seed_strength=1.0,
        age_profile=None,
        seeding_start="2020-02-28",
        seeding_end="2020-03-02",
    ):
        self.seed_strength = seed_strength
        self.age_profile = age_profile
        self.seeding_start = seeding_start
        self.seeding_end = seeding_end

    @classmethod
    def from_parameters(cls, parameters: dict):
        infection_parameters = parameters["infection"]
        seed_strength = infection_parameters.get("seed_strength", None)
        age_profile = infection_parameters.get("age_profile", None)
        seeding_start = infection_parameters.get("seeding_start", None)
        seeding_end = infection_parameters.get("seeding_end", None)
        return cls(
            seed_strength=seed_strength,
            age_profile=age_profile,
            seeding_start=seeding_start,
            seeding_end=seeding_end
        )

    def make_infection_seed(self, infection_selector, world):
        oc = Observed2Cases.from_file(
            health_index_generator=infection_selector.health_index_generator,
            smoothing=True,
        )
        daily_cases_per_region = oc.get_regional_latent_cases()
        daily_cases_per_super_area = oc.convert_regional_cases_to_super_area(
            daily_cases_per_region, dates=[self.seeding_start, self.seeding_end]
        )

        infection_seed = InfectionSeed(
            world=world,
            infection_selector=infection_selector,
            daily_super_area_cases=daily_cases_per_super_area,
            seed_strength=self.seed_strength,
            age_profile=self.age_profile,
        )
        return infection_seed
