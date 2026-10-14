#!/bin/bash
# Copied from campaign 0070 (scripts/el9.sh) so that this application needs
# nothing from the campaign repository.
# Run a command inside the CMS EL9 image WITHOUT the cmssw-el9 wrapper.
#
# Why not the wrapper: it force-mounts /asap3 /beegfs /gpfs /nfs /pnfs when
# they exist.  On the CBE compute nodes /beegfs does exist, and binding it
# wedges the apptainer setup in uninterruptible sleep -- the container never
# starts and the payload never runs.  The p.quality array lost 29 of its 32
# tasks that way on 2026-10-02, each sitting the full eight hours doing
# nothing.  timeout cannot clear it: a process in D state ignores SIGTERM.
# The login node has no /beegfs, which is why interactive work kept working
# while every batch task hung.
#
# What the wrapper DOES do that matters, and is reproduced here: it binds the
# grid credentials.  Without /etc/grid-security the GSI client inside the
# container has no CA certificates, every EOS open comes back "Operation not
# permitted", and the job then segfaults in the ROOT exception handler rather
# than reporting a credential problem.
set -u
IMG="${DAS_EL9_IMAGE:-/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmssw/el9:x86_64}"
[ -d "$IMG" ] || { echo "el9.sh: image not found: $IMG" >&2; exit 2; }

binds=(-B /cvmfs -B /eos -B /scratch-cbe -B /scratch -B /users)
# grid credentials and CA trust, as the wrapper does
for d in /etc/grid-security /etc/vomses /etc/pki/ca-trust /run/user /var/run/user; do
    [ -e "$d" ] && binds+=(-B "$d")
done

# The payload reads EOS over xrootd and nothing else sets this.
if [ -z "${X509_USER_PROXY:-}" ] && [ -r "$HOME/.private/.proxy" ]; then
    export X509_USER_PROXY="$HOME/.private/.proxy"
fi

exec apptainer -s exec "${binds[@]}" "$IMG" "$@"
