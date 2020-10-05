import yaml
import sys

from june.policy import Policies


def str_to_class(classname):
    return getattr(sys.modules["june.policy"], classname)


class PolicySetter:
    def __init__(self, policies_baseline: dict):
        self.policies_baseline = policies_baseline

    def build_config_for_lockdown(
        self,
        soft_lockdown_date: str,
        hard_lockdown_date: str,
        lockdown_ratio: int,
        hard_lockdown_policy_parameters: dict,
    ):
        for policy, policy_parameters in hard_lockdown_policy_parameters:



    def make_policies(self, policies_to_modify):
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
                    if policy in policies_to_modify:
                        if policy_i in policies_to_modify[policy]:
                            tomodify = policies_to_modify[policy][policy_i]
                            for parameter, parameter_value in tomodify.items():
                                policy_data_i[parameter] = parameter_value
                    policies.append(str_to_class(camel_case_key)(**policy_data_i))
            else:
                if policy in policies_to_modify:
                    tomodify = policies_to_modify[policy]
                    for parameter, parameter_value in tomodify.items():
                        policy_data[parameter] = parameter_value
                policies.append(str_to_class(camel_case_key)(**policy_data))
        return Policies(policies=policies)
