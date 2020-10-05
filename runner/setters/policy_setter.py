import yaml
import sys
from datetime import datetime
from copy import deepcopy

from june.policy import Policies


def str_to_class(classname):
    return getattr(sys.modules["june.policy"], classname)


class PolicySetter:
    def __init__(self, policies_baseline: dict):
        self.policies_baseline = policies_baseline

    def _set_lockdown_policy(
        self,
        ret,
        policy_type,
        policy_number,
        policy_data,
        soft_lockdown_date,
        hard_lockdown_date,
        hard_lockdown_policy_parameters,
        lockdown_ratio,
    ):
        if policy_data["start_time"] == soft_lockdown_date:
            if policy_type == "social_distancing":
                overall_beta_factor = hard_lockdown_policy_parameters[
                    "social_distancing"
                ]["overall_beta_factor"]
                for group in policy_data["beta_factors"]:
                    if group == "household":
                        continue
                    if policy_number is None:
                        ret[policy_type]["beta_factors"][group] = (
                            overall_beta_factor
                            + (1 - overall_beta_factor) * lockdown_ratio
                        )
                    else:
                        ret[policy_type][policy_number]["beta_factors"][group] = (
                            overall_beta_factor
                            + (1 - overall_beta_factor) * lockdown_ratio
                        )
            else:
                for parameter, value in hard_lockdown_policy_parameters[
                    policy_type
                ].items():
                    if policy_number is None:
                        ret[policy_type][parameter] = lockdown_ratio * value
                    else:
                        ret[policy_type][policy_number][parameter] = (
                            lockdown_ratio * value
                        )
        elif policy_data["start_time"] == hard_lockdown_date:
            if policy_type == "social_distancing":
                overall_beta_factor = hard_lockdown_policy_parameters[
                    "social_distancing"
                ]["overall_beta_factor"]
                for group in policy_data["beta_factors"]:
                    if group == "household":
                        continue
                    if policy_number is None:
                        ret[policy_type]["beta_factors"][group] = overall_beta_factor
                    else:
                        ret[policy_type][policy_number]["beta_factors"][
                            group
                        ] = overall_beta_factor
            else:
                for parameter, value in hard_lockdown_policy_parameters[
                    policy_type
                ].items():
                    if policy_number is None:
                        ret[policy_type][parameter] = value
                    else:
                        ret[policy_type][policy_number][parameter] = value

    def build_config_for_lockdown(
        self,
        soft_lockdown_date: str,
        hard_lockdown_date: str,
        lockdown_ratio: int,
        hard_lockdown_policy_parameters: dict,
    ):
        soft_lockdown_date = datetime.strptime(soft_lockdown_date, "%Y-%m-%d").date()
        hard_lockdown_date = datetime.strptime(hard_lockdown_date, "%Y-%m-%d").date()
        ret = deepcopy(self.policies_baseline)
        for policy_type in self.policies_baseline:
            if policy_type in hard_lockdown_policy_parameters:
                if "start_time" in self.policies_baseline[policy_type]:
                    policy_data = self.policies_baseline[policy_type]
                    self._set_lockdown_policy(
                        ret=ret,
                        policy_number=None,
                        policy_type=policy_type,
                        policy_data=policy_data,
                        soft_lockdown_date=soft_lockdown_date,
                        hard_lockdown_date=hard_lockdown_date,
                        hard_lockdown_policy_parameters=hard_lockdown_policy_parameters,
                        lockdown_ratio=lockdown_ratio,
                    )
                else:
                    for policy_i, policy_data in self.policies_baseline[
                        policy_type
                    ].items():
                        self._set_lockdown_policy(
                            ret=ret,
                            policy_number=policy_i,
                            policy_type=policy_type,
                            policy_data=policy_data,
                            soft_lockdown_date=soft_lockdown_date,
                            hard_lockdown_date=hard_lockdown_date,
                            hard_lockdown_policy_parameters=hard_lockdown_policy_parameters,
                            lockdown_ratio=lockdown_ratio,
                        )
        return ret

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
                            policy_data_modified = deepcopy(policy_data_i)
                            tomodify = policies_to_modify[policy][policy_i]
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
                if policy in policies_to_modify:
                    policy_data_modified = deepcopy(policy_data)
                    tomodify = policies_to_modify[policy]
                    for parameter, parameter_value in tomodify.items():
                        policy_data_modified[parameter] = parameter_value
                else:
                    policy_data_modified = policy_data
                policies.append(str_to_class(camel_case_key)(**policy_data_modified))
        return Policies(policies=policies)
