#!/usr/bin/env bash
set -euo pipefail

sudo docker ps -a --filter name=dsv4-256k-measure-tutti
sudo docker rm -f dsv4-256k-measure-tutti || true
sudo docker ps -a --filter name=dsv4-256k-measure-tutti
