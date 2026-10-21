import re

def instantiate_files(tmpl_path, config_paths, path_value):
    config = config_paths.copy()
    for term, path in config.items():
        if callable(path):
            path = path(path_value)
        with open(path, "r") as file:
            config[term] = file.read()

    return instantiate_values(tmpl_path, config)

def instantiate_values(tmpl_path, config):
    with open(tmpl_path, "r") as fh:
        template = fh.read()

    for term, value in config.items():
        template = template.replace(term, value)

    return template