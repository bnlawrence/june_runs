from june.infection import HealthIndexGenerator
from june_runs.setters import HealthIndexSetter

def test__asymptomatic_ratio():
    his = HealthIndexSetter(asymptomatic_ratio=0.6)
    hi = his.make_health_index()
    assert isinstance(hi, HealthIndexGenerator)
    assert hi.asymptomatic_ratio == 0.6
