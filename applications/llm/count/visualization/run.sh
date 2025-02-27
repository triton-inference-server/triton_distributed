#!/bin/bash

# FIXME: Use docker compose

# Prometheus
docker run --rm -d -p 9090:9090 --network host --name prometheus -v ./prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus:latest

# Grafana
docker run --rm -d -p 3000:3000 --network host --name=grafana -v ./grafana.json:/etc/grafana/provisioning/dashboards/llm-worker-dashboard.json -v ./grafana-datasources.yml:/etc/grafana/provisioning/datasources/datasources.yml -v ./grafana-dashboard-providers.yml:/etc/grafana/provisioning/dashboards/dashboard-providers.yml grafana/grafana-enterprise
