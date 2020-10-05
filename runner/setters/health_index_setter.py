from june.infection import HealthIndexGenerator


class HealthIndexSetter:
    def __init__(self, asymptomatic_ratio=0.2):
        self.asymptomatic_ratio = asymptomatic_ratio

    def make_health_index(self):
        return HealthIndexGenerator.from_file(
            asymptomatic_ratio=self.asymptomatic_ratio
        )
