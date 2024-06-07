# Mage learning project

This repository contains the personal work done to learn Mage, as part of the 2024 cohort of the MLOps Zoomcap organized by the DataTalks.Club community.The repo is a fork of the template provided by the Mage team at https://github.com/mage-ai/mlops.

## Start mage

1. Change directory into the cloned repo:

   ```
   cd mlops
   ```

2. Launch Mage and the database service (PostgreSQL):

   ```
   ./scripts/start.sh
   ```
   The script sets some environmental variables and launches docker compose, which builds and starts the containers for Mage and a PostgreSQL database.

3. Open [`http://localhost:6789`](http://localhost:6789) in your browser to access Mage.
