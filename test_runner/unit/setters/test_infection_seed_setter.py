from datetime import datetime

from june.infection_seed import InfectionSeed
from june.infection import HealthIndexGenerator, InfectionSelector
from june_runs.setters import InfectionSeedSetter


def test__set_parameters():
    age_profile = {"0-20": 0.5, "20-100": 0.5}
    iss = InfectionSeedSetter(seed_strength=5, age_profile=age_profile)
    hi = HealthIndexGenerator.from_file()
    infection_selector = InfectionSelector.from_file(
        health_index_generator=hi
    )
    infection_seed = iss.make_infection_seed(infection_selector=infection_selector, world=None)
    assert isinstance(infection_seed, InfectionSeed)
    assert infection_seed.seed_strength == 5
    assert infection_seed.age_profile == age_profile


def test__set_dates_of_seeding():
    iss = InfectionSeedSetter(seeding_start="2020-03-01", seeding_end="2020-03-03")
    hi = HealthIndexGenerator.from_file()
    infection_selector = InfectionSelector.from_file(
        health_index_generator=hi
    )
    infection_seed = iss.make_infection_seed(infection_selector=infection_selector, world=None)
    assert infection_seed.min_date == datetime.strptime("2020-03-01", "%Y-%m-%d")
    assert infection_seed.max_date == datetime.strptime("2020-03-03", "%Y-%m-%d")
