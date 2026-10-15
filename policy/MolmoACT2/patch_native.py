"""Apply the small Marvin extension to the separately installed AllenAI LeRobot.

Fails on unknown source layouts. Re-running is idempotent. The native backbone,
trainer and processors remain responsible for training and serialization.
"""
from pathlib import Path
import argparse

MARKER = '# DexDojo Marvin action width support'


def patch(root):
    folder = Path(root) / 'src/lerobot/policies/molmoact2'
    config = folder / 'configuration_molmoact2.py'
    model = folder / 'modeling_molmoact2.py'
    changes = []
    for path in (config, model):
        source = path.read_text()
        if MARKER in source:
            continue
        if path == config:
            old = 'if self.expected_max_action_dim != 32:'
            assert old in source, f'Unknown MolmoACT2 source: {path}'
            source = source.replace(old, 'if self.expected_max_action_dim not in (32, 54):')
        else:
            old = '        self.model = MolmoAct2ForConditionalGeneration.from_pretrained('
            assert old in source and 'if hf_max_action_dim != 32:' in source, f'Unknown MolmoACT2 source: {path}'
            source = source.replace(old, '''        adapt_robot = int(hf_config.max_action_dim) != int(self.config.expected_max_action_dim)
        hf_config.max_action_dim = int(self.config.expected_max_action_dim)
        self.model = MolmoAct2ForConditionalGeneration.from_pretrained(''')
            source = source.replace('            config=hf_config,', '            config=hf_config,\n            ignore_mismatched_sizes=adapt_robot,')
            source = source.replace('if hf_max_action_dim != 32:', 'if hf_max_action_dim not in (32, 54):')
            # The additional strict load audits all backbone shapes; only 32->54
            # action projections are left at their freshly initialized values.
            source = source.replace('model.load_state_dict(state_dict, strict=False)',
                                    'model.load_state_dict(_marvin_projection_weights(model, state_dict), strict=False)')
            source = source.replace('model.load_state_dict(state_dict, strict=True)',
                                    'model.load_state_dict(_marvin_projection_weights(model, state_dict), strict=True)')
            source += '''\n\ndef _marvin_projection_weights(model, weights):
    own = model.state_dict()
    result = dict(weights)
    for key, value in weights.items():
        if key in own and value.shape != own[key].shape:
            changes = [(a,b) for a,b in zip(value.shape, own[key].shape) if a != b]
            if "action" not in key or value.ndim != own[key].ndim or not changes or any(pair != (32,54) for pair in changes):
                raise ValueError(f"Unexpected checkpoint shape mismatch: {key}")
            result[key] = own[key].detach().clone()
    return result
'''
        changes.append((path, source + '\n' + MARKER + '\n'))
    for path, source in changes:
        compile(source, str(path), 'exec')
    for path, source in changes:
        path.write_text(source)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('lerobot_root')
    patch(parser.parse_args().lerobot_root)
