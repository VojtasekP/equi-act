import subprocess

# Download dataset
subprocess.run(["wget", "-nc", "http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip"])

# Unzip dataset
subprocess.run(["unzip", "-n", "mnist_rotation_new.zip", "-d", "mnist_rotation_new"])
