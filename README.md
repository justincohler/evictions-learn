# evictions-learn
CAPP-30254 Final Project Repository

[![Build Status](https://travis-ci.org/justincohler/evictions-learn.svg?branch=master)](https://travis-ci.org/justincohler/evictions-learn)

## About
This research uses data from The Eviction Lab at Princeton University, a project directed by Matthew Desmond and designed by Ashley Gromis, Lavar Edmonds, James Hendrickson, Katie Krywokulski, Lillian Leung, and Adam Porton. The Eviction Lab is funded by the JPB, Gates, and Ford Foundations as well as the Chan Zuckerberg Initiative. More information is found at evictionlab.org.

[Project Proposal](https://docs.google.com/document/d/1Vsq1RUL8fU5U8FO2KtszmsdIYEhmlUN5twY_q0k-RvQ/edit?usp=sharing)

## Setup
* Fill in the template in src/set_env.sh with your database credentials and host details.
* Ensure the following environment variables are set (e.g. `EXPORT DB_HOST=dev.yourenv.com`):
  * `DB_HOST`
  * `DB_PORT`
  * `DB_USER`
  * `DB_PASSWORD`
* Run `pip install -r requirements.txt` in the top level of the repository.
* Run `python setup.py install`.
* Run `pytest -s` to run the test suite. If the tests pass, then you're successfully connected to the database!
