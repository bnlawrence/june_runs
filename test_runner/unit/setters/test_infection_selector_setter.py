from runner.setters import InfectionSelectorSetter
from june.infection import InfectionSelector, HealthIndexGenerator
from june.infection.transmission import TransmissionConstant, TransmissionGamma
from june.infection.transmission_xnexp import TransmissionXNExp
from june.demography import Person

def test__infectivity_profile():
    hi = HealthIndexGenerator.from_file()
    iss = InfectionSelectorSetter(infectivity_profile="xnexp")
    infection_selector = iss.make_infection_selector(hi)
    assert infection_selector.health_index_generator == hi
    assert isinstance(infection_selector, InfectionSelector)
    person = Person.from_attributes()
    infection_selector.infect_person_at_time(person, 0)
    assert person.infection
    assert isinstance(person.infection.transmission, TransmissionXNExp)

    iss = InfectionSelectorSetter(infectivity_profile="nature")
    infection_selector = iss.make_infection_selector(hi)
    person = Person.from_attributes()
    infection_selector.infect_person_at_time(person, 0)
    assert isinstance(person.infection.transmission, TransmissionGamma)

    iss = InfectionSelectorSetter(infectivity_profile="correction_nature")
    infection_selector = iss.make_infection_selector(hi)
    person = Person.from_attributes()
    infection_selector.infect_person_at_time(person, 0)
    assert isinstance(person.infection.transmission, TransmissionGamma)

    iss = InfectionSelectorSetter(infectivity_profile="constant")
    infection_selector = iss.make_infection_selector(hi)
    person = Person.from_attributes()
    infection_selector.infect_person_at_time(person, 0)
    assert isinstance(person.infection.transmission, TransmissionConstant)
