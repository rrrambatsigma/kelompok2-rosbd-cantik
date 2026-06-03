#!/bin/bash
prefect deployment build /flows/main_flow.py:main_flow -n opensky-deployment
prefect deployment apply main_flow-deployment.yaml