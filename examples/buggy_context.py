import subprocess


def load_expression(path, command):
    handle = open(path, "r", encoding="utf-8")
    expression = handle.read()
    result = eval(expression)
    subprocess.run(command, shell=True)
    return result
