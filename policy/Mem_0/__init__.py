"""Load inference frameworks only in the policy process, never for deploy imports."""
def __getattr__(name):
    if name == "Model":
        from .model import Model
        return Model
    if name in ("eval_one_episode", "eval_one_episode_batch"):
        from importlib import import_module
        return getattr(import_module(__name__ + ".deploy"), name)
    raise AttributeError(name)


def get_model(deploy_cfg):
    from importlib import import_module
    module = import_module(__name__ + ".model")
    return getattr(module, "create_model", module.Model)(deploy_cfg)
