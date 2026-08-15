"""YAML configuration loading with small, explicit experiment overrides."""

from copy import deepcopy
from pathlib import Path

import yaml


def _deep_update(base, overrides):
    result = deepcopy(base)
    for key, value in overrides.items():
        if (isinstance(value, dict)
                and isinstance(result.get(key), dict)):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path, _stack=()):
    path = Path(path).resolve()
    if path in _stack:
        chain = ' -> '.join(map(str, (*_stack, path)))
        raise ValueError(f'cyclic base_config chain: {chain}')
    with path.open('r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f'configuration must be a mapping: {path}')
    base_name = config.pop('base_config', None)
    if base_name is None:
        return config
    base = load_config(path.parent / base_name, (*_stack, path))
    return _deep_update(base, config)
