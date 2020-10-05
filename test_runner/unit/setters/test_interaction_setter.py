import pytest

from june.demography import Person, Population
from june.interaction import Interaction
from runner.setters import InteractionSetter

@pytest.fixture(name='people')
def make_pop():
    people = Population([])
    for i in range(100):
        people.add(Person.from_attributes(age=i))
    return people

def test__change_susceptibility(people):
    susceptibilities = {"0-20": 0.5, "20-100": 1.0}
    interaction_setter = InteractionSetter(
        susceptibilities_by_age=susceptibilities, population=people
    )
    interaction = interaction_setter.make_interaction()
    assert isinstance(interaction, Interaction)
    for person in people:
        if person.age < 20:
            assert person.susceptibility == 0.5
        else:
            assert person.susceptibility == 1.0


def test__change_betas(people):
    betas = {
        "care_home": 0.5,
        "household": 1.0,
        "company": 0.25,
        "city_transport": 2.0,
        "pub": 0.5,
    }
    interaction_setter = InteractionSetter(betas=betas, population=people)
    interaction = interaction_setter.make_interaction()
    for beta, value in betas.items():
        assert interaction.beta[beta] == value


def test__non_existing_beta_errors(people):
    betas = {"white_house": 1000}
    with pytest.raises(Exception) as execinfo:
        interaction_setter = InteractionSetter(betas=betas, population=people)
        interaction = interaction_setter.make_interaction()
        assert (
            execinfo.value.args[0]
            == f"Trying to change a beta for a non-existing group."
        )


def test__change_alpha_physical(people):
    alpha_physical = 3
    interaction_setter = InteractionSetter(alpha_physical=alpha_physical, population=people)
    interaction = interaction_setter.make_interaction()
    assert interaction.alpha_physical == alpha_physical
