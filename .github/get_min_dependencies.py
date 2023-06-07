"""This script fetches minimum dependencies of cleanlab package and writes them to the file requirements-min.txt"""
import json


if __name__ == "__main__":
    with open("./deps.json", "r") as f:
        deps = json.load(f)

    for package in deps:
        if package["package"]["package_name"] == "cleanlab":
            for dep in package["dependencies"]:
                req_version = dep["required_version"]
                with open("requirements-min.txt", "a") as f:
                    if req_version.startswith(">="):
                        f.write(f"{dep['package_name']}=={req_version[2:]}\n")
