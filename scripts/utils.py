import os


def make_pylidc_config(path: str) -> None:
    config_path = os.path.expanduser("~/.pylidcrc")
    expanded_path = os.path.expanduser(path)
    with open(config_path, "w") as f:
        f.write(
            f"[dicom]\npath = {os.path.join(expanded_path, 'data/LIDC-IDRI')}\nwarn = True\n"
        )
