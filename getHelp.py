import subprocess, os

commands = ["ls"]
for command in commands:
    try:
        subprocess.run([command, "--help"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        os.system(f"{command} --help > help-docs/{command}.txt")
    except subprocess.CalledProcessError as e:
        print(f"{command} didn't work")

