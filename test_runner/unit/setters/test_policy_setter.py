import pytest
import yaml

from datetime import datetime
from pathlib import Path

from june.policy import Hospitalisation, CloseCompanies, Quarantine, Policies
from runner.setters import PolicySetter

test_config = Path(__file__).parent / "test_policy_config.yaml"


def check_policy_change(changed_policy, base_policy, parameters_changed: dict):
    for param1, param2 in zip(changed_policy.__dict__, base_policy.__dict__):
        if param1 in parameters_changed:
            attribute1 = getattr(changed_policy, param1)
            attribute2 = getattr(base_policy, param2)
            assert getattr(changed_policy, param1) == parameters_changed[param1]
        else:
            attribute1 = getattr(changed_policy, param1)
            attribute2 = getattr(base_policy, param2)
            if type(attribute1) == dict:
                assert type(attribute2) == dict
                for key1, value1 in attribute1.items():
                    assert attribute2[key1] == value1
            else:
                assert attribute1 == attribute2


class TestChangePoliciesParameters:
    @pytest.fixture(name="policy_setter", scope="module")
    def make_ps(self):
        with open(test_config) as f:
            baseline_config = yaml.load(f, Loader=yaml.FullLoader)
        policy_setter = PolicySetter(policies_baseline=baseline_config)
        return policy_setter

    @pytest.fixture(name="policies", scope="module")
    def make_policies(self, policy_setter):
        policies_to_modify = {
            "hospitalisation": {"probability_of_care_home_resident_admission": 0.4},
            "quarantine": {
                1: {"n_days": 10, "household_compliance": 0.5},
                2: {"compliance": 0.2},
            },
            "close_companies": {3: {"avoid_work_probability": 0.20}},
        }
        policies = policy_setter.make_policies(policies_to_modify)
        return policies

    @pytest.fixture(name="base", scope="module")
    def base(self):
        return Policies.from_file(test_config)

    def test__hospitalisation_parameters(self, policies, base):
        for policy1, policy2 in zip(policies, base):
            if isinstance(policy1, Hospitalisation):
                assert isinstance(policy2, Hospitalisation)
                check_policy_change(
                    policy1,
                    policy2,
                    {"probability_of_care_home_resident_admission": 0.4},
                )

    def test__quarantine_parameters(self, policies, base):
        for policy1, policy2 in zip(policies, base):
            if isinstance(policy1, Quarantine):
                assert isinstance(policy2, Quarantine)
                if policy1.start_time == datetime.strptime("2020-03-16", "%Y-%m-%d"):
                    check_policy_change(
                        policy1, policy2, {"n_days": 10, "household_compliance": 0.5}
                    )
                elif policy1.start_time == datetime.strptime("2020-03-24", "%Y-%m-%d"):
                    check_policy_change(policy1, policy2, {"compliance": 0.2})
                else:
                    check_policy_change(policy1, policy2, {})

    def test__close_companies_parameters(self, policies, base):
        for policy1, policy2 in zip(policies, base):
            if isinstance(policy1, CloseCompanies):
                assert isinstance(policy2, CloseCompanies)
                if policy1.start_time == datetime.strptime("2020-03-27", "%Y-%m-%d"):
                    check_policy_change(
                        policy1, policy2, {"avoid_work_probability": 0.2}
                    )
                else:
                    check_policy_change(policy1, policy2, {})

    # TODO: extend tests to all policy types.

class TestMakeLockdownParameters:
    def setup(self):
        soft_lockdown_date = "2020-03-16"
