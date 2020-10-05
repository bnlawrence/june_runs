from june.infection import HealthIndexGenerator


class HealthIndexSetter:
    def __init__(self, asymptomatic_ratio=0.2):
        self.asymptomatic_ratio = asymptomatic_ratio

    @classmethod
    def from_parameters(cls, parameters: dict):
        asymptomatic_ratio = parameters.get("asymptomatic_ratio", None)
        return cls(asymptomatic_ratio=asymptomatic_ratio)

    def make_health_index(self):
        return HealthIndexGenerator.from_file(
            asymptomatic_ratio=self.asymptomatic_ratio
        )
