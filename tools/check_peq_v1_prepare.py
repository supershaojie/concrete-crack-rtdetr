"""Regression checks for YAML/JSON data-config identity; no dataset reads or training."""
from copy import deepcopy
from peq_v1_common import *
from peq_v1_runtime import verify_data_config_semantics
from ultralytics.utils import YAML


def config_semantics_checks():
    baseline_path=ROOT/"docs/peq_v1/dataset_identity.json"
    baseline_hash=sha256(baseline_path)
    server=YAML.load(ROOT/"docs/c19_lif_v1/c2_data.yaml")
    audited=read_json(baseline_path)["config_semantics"]
    raw_differences=[key for key in sorted(set(server)|set(audited))
                     if key!="path" and server.get(key)!=audited.get(key)]
    require(raw_differences==["names"],"Actual audit has additional differences")
    require(server["names"]=={0:"crack"} and audited["names"]=={"0":"crack"},"Reproduction changed")
    verify_data_config_semantics(server,audited)
    passed=["actual_mother_yaml_vs_audit_json"]
    migrated=deepcopy(server);migrated["path"]="/another/data/root"
    verify_data_config_semantics(migrated,audited)
    passed.append("only_top_level_path_migration")
    reordered={key:server[key] for key in reversed(server)}
    verify_data_config_semantics(reordered,audited)
    passed.append("mapping_order_ignored")
    cases=[]
    for field,value in (("names",{0:"other_crack"}),("names",{1:"crack"}),
                        ("names",["crack"]),("train","images/other_train"),
                        ("val","images/other_val"),("test","images/other_test"),
                        ("extra_setting",True)):
        changed=deepcopy(server);changed[field]=value
        cases.append((f"changed_{field}_{len(cases)}",changed,audited,field))
    missing=deepcopy(server);missing.pop("val")
    cases.append(("missing_val",missing,audited,"val"))
    extended_server=dict(server,options={"enabled":True,"path":"nested_original"},nc=1)
    extended_audit=dict(audited,options={"enabled":True,"path":"nested_original"},nc=1)
    verify_data_config_semantics(extended_server,extended_audit)
    passed.append("unchanged_additional_fields")
    for field,value in (("options",{"enabled":False,"path":"nested_original"}),
                        ("options",{"enabled":True,"path":"nested_changed"}),("nc",True),("nc",1.0)):
        changed=deepcopy(extended_server);changed[field]=value
        cases.append((f"changed_{field}_{len(cases)}",changed,extended_audit,field))
    failures=[]
    for name,current,expected,field in cases:
        try:
            verify_data_config_semantics(current,expected)
        except RuntimeError as error:
            message=str(error)
            require(f"{field}: server=" in message and "; audited=" in message,
                    f"Missing field or both values in failure: {message}")
            failures.append(dict(case=name,error=message))
        else:
            raise AssertionError(f"Real config difference accepted: {name}")
    collision=deepcopy(server);collision["names"]={0:"wrong","0":"crack"}
    try:
        verify_data_config_semantics(collision,audited)
    except RuntimeError as error:
        require("Duplicate JSON field" in str(error),"Ambiguous JSON key diagnostic missing")
        failures.append(dict(case="duplicate_normalized_class_key",error=str(error)))
    else:
        raise AssertionError("JSON normalization silently dropped a class key")
    require(sha256(baseline_path)==baseline_hash,"Audit baseline changed")
    return dict(status="PASS",raw_non_path_differences=raw_differences,
                server_names=repr(server["names"]),audited_names=repr(audited["names"]),
                representation_only=True,positive_cases=passed,rejected_cases=failures,
                baseline_sha256=baseline_hash,audit_baseline_unchanged=True,
                scope="Config comparison only; no dataset mutation, full prepare or training")


if __name__=="__main__":
    result=config_semantics_checks()
    write_json(OUT/"prepare_fix_validation.json",result)
    print(json_bytes(result).decode())
