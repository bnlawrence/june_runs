import yaml
import sys
from datetime import datetime
from copy import deepcopy

from june.policy import Policies


def str_to_class(classname):
    return getattr(sys.modules["june.policy"], classname)


class PolicySetter:
    def __init__(self, policies_baseline: dict, policies_to_modify: dict):
        self.policies_baseline = policies_baseline
        self.policies_to_modify = policies_to_modify

    @classmethod
    def from_parameters(cls, policies_to_modify: dict, baseline_policy_path: str):
        with open(baseline_policy_path, "r") as f:
            policies_baseline = yaml.load(f, Loader=yaml.FullLoader)
        return cls(
            policies_baseline=policies_baseline, policies_to_modify=policies_to_modify
        )

    def make_policies(self):
        policies = []
        for policy, policy_data in self.policies_baseline.items():
            camel_case_key = "".join(x.capitalize() or "_" for x in policy.split("_"))
            if "start_time" not in policy_data:
                for policy_i, policy_data_i in policy_data.items():
                    if (
                        "start_time" not in policy_data_i.keys()
                        or "end_time" not in policy_data_i.keys()
                    ):
                        raise ValueError("policy config file not valid.")
                    if policy in self.policies_to_modify:
                        if str(policy_i) in self.policies_to_modify[policy]:
                            policy_data_modified = deepcopy(policy_data_i)
                            tomodify = self.policies_to_modify[policy][str(policy_i)]
                            for parameter, parameter_value in tomodify.items():
                                policy_data_modified[parameter] = parameter_value
                        else:
                            policy_data_modified = policy_data_i
                    else:
                        policy_data_modified = policy_data_i
                    policies.append(
                        str_to_class(camel_case_key)(**policy_data_modified)
                    )
            else:
                if policy in self.policies_to_modify:
                    policy_data_modified = deepcopy(policy_data)
                    tomodify = self.policies_to_modify[policy]
                    for parameter, parameter_value in tomodify.items():
                        policy_data_modified[parameter] = parameter_value
                else:
                    policy_data_modified = policy_data
                policies.append(str_to_class(camel_case_key)(**policy_data_modified))
        return Policies(policies=policies)
