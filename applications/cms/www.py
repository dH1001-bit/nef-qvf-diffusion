"""Publish plots to the CERN www area from CBE via plain xrdcp.

Usage, from the repo root on CBE:

    python -m applications.cms.www <subdir> <files...>

or from another script:

    from applications.cms.www import publish
    publish(["plot.png", "plot.pdf"], "quicklook")

Files land under /eos/user/s/schoef/www/nef-qvf-diffusion/<base>/<subdir> (base: TT2l-study, or --base JER)
and are served at
https://schoef.web.cern.ch/schoef/nef-qvf-diffusion/TT2l-study/<subdir>/;
an index.php is copied alongside so the directory is browsable.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

EOS = "root://eosuser.cern.ch"
BASE = "/eos/user/s/schoef/www/nef-qvf-diffusion/TT2l-study"
URL = "https://schoef.web.cern.ch/schoef/nef-qvf-diffusion/TT2l-study"
INDEX_PHP = (
    "/users/robert.schoefbeck/CMS/ML/HEPHY-uncertainty/common/scripts/php/index.php"
)
PROXY = "/users/robert.schoefbeck/.private/.proxy"


def _environment() -> dict:
    environment = dict(os.environ)
    environment.setdefault("X509_USER_PROXY", PROXY)
    return environment


def publish(paths, subdir: str) -> list[str]:
    """Copy the files one by one to TT2l-study/<subdir>; return the URLs."""

    destination = f"{BASE}/{subdir}" if subdir else BASE
    environment = _environment()
    subprocess.run(["xrdfs", EOS, "mkdir", "-p", destination], check=True, env=environment)
    urls = []
    copies = [str(p) for p in paths]
    if Path(INDEX_PHP).exists():
        copies.append(INDEX_PHP)
    for path in copies:
        name = Path(path).name
        subprocess.run(
            ["xrdcp", "-f", "--nopbar", path,
             f"{EOS}//{destination.lstrip('/')}/{name}"],
            check=True,
            env=environment,
        )
        if name != "index.php":
            urls.append(f"{URL}/{subdir}/{name}" if subdir else f"{URL}/{name}")
    for url in urls:
        print(url)
    return urls


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit("usage: python -m applications.cms.www [--base JER] <subdir> <files...>")
    args = sys.argv[1:]
    base = "TT2l-study"
    if args[0] == "--base":
        base, args = args[1], args[2:]
    publish(args[1:], args[0], base=base)


if __name__ == "__main__":
    main()
