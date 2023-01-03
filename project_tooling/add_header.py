# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# Usage:
#
# python project_tooling/add_header.py
#
# Running this script will:
# 1. Remove OLD_HEADER from any targeted file that possesses that header
# 2. Add NEW_HEADER to the beginning of all targeted files
#
# See __main__ to see which directories/files are targeted by this script.
# Modify this section in the event that a new directory of .py files is
# added to the repo, which does not fall under any of the current directories.

import os
import os.path as path
from pathlib import Path
from typing import Iterable, Union

OLD_HEADER = """# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT\n"""

NEW_HEADER = """# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT\n"""

EXCLUDED = {"_version.py", "versioneer.py"}

OLD_HEADER = OLD_HEADER.splitlines(keepends=True)
NEW_HEADER = NEW_HEADER.splitlines(keepends=True)


def is_safe_dir(path: Path):
    path = path.resolve()
    if not Path.cwd() in path.parents:
        raise ValueError(
            f"Dangerous! You are running a script that can overwrite files in an unexpected directory: {path}"
        )


def get_src_files(dirname: Union[str, Path]) -> Iterable[Path]:
    dirname = Path(dirname)

    if dirname.is_file():
        if dirname.name.endswith(".py") and dirname.name not in EXCLUDED:
            yield dirname

    else:
        for cur, _, files in os.walk(dirname):
            cur = Path(cur)

            if any(p.startswith((".", "__")) for p in cur.parts):
                # exclude hidden/meta dirs
                continue

            for f in files:
                if f in EXCLUDED:
                    continue
                if f.endswith(".py"):
                    yield Path(path.join(cur, f))


def add_headers(files: Iterable[Path]):
    for file_ in files:
        is_safe_dir(file_)

        with file_.open("r") as f:
            contents = f.readlines()

        if contents[: len(OLD_HEADER)] == OLD_HEADER:
            contents = contents[len(OLD_HEADER) :]

        if contents[: len(NEW_HEADER)] != NEW_HEADER:
            contents = NEW_HEADER + contents

        with file_.open("w") as f:
            f.writelines(contents)


if __name__ == "__main__":
    add_headers(get_src_files("./setup.py"))
    add_headers(get_src_files("./src/rai_toolbox/"))
    add_headers(get_src_files("./tests/"))
    add_headers(get_src_files("./experiments/"))
