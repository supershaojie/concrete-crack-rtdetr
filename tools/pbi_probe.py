"""Temporary PBI taps around the unchanged parent's candidate-aware probe.

The functional Wo convolution is deliberately observed at the PBI output rather
than via a Conv2d hook, which would never run. All hooks disappear before callers
may copy or serialize models.
"""
from c19_lif_v1_probe import capture as parent_capture, targets


def capture(model, image, batch=None, fixed_ids=None):
    module = getattr(model.model[17], "pbi", None)
    if module is None:
        return parent_capture(model, image, batch, fixed_ids)
    records, handles, calls = {}, [], []

    def save(name, value):
        records[name] = value.detach().cpu().clone()

    def branch(mod, args, value):
        calls.append(1)
        save("pbi_input", args[0])
        save("pbi_output", value)
        save("pbi_residual", value.float() - args[0].float())

    handles.append(module.register_forward_hook(branch))
    handles.append(module.W1.register_forward_hook(lambda mod, args, value: save("pbi_u", value)))
    handles.append(module.W2.register_forward_hook(lambda mod, args, value: save("pbi_v", value)))
    try:
        value, parent = parent_capture(model, image, batch, fixed_ids)
        if len(calls) != 1:
            raise RuntimeError("PBI must execute exactly once in each native forward")
        parent.update(records)
        return value, parent
    finally:
        for handle in handles:
            handle.remove()
