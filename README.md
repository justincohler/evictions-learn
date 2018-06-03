# evictions-learn
CAPP-30254 Final Project Repository

[![Build Status](https://travis-ci.org/justincohler/evictions-learn.svg?branch=master)](https://travis-ci.org/justincohler/evictions-learn)

## About
This research was conducted by Claire Herdeman, Alena Stern, and Justin Cohler for the Machine Learning and Public Policy course in the Masters of Science in Computational Analysis and Public Policy program at the University of Chicago. This research uses data from The Eviction Lab at Princeton University, a project directed by Matthew Desmond and designed by Ashley Gromis, Lavar Edmonds, James Hendrickson, Katie Krywokulski, Lillian Leung, and Adam Porton. The Eviction Lab is funded by the JPB, Gates, and Ford Foundations as well as the Chan Zuckerberg Initiative. More information is found at evictionlab.org. The authors owe a debt of gratitude to The Eviction Lab for providing the inspiration and data for this project. 

[Project Proposal](https://docs.google.com/document/d/1Vsq1RUL8fU5U8FO2KtszmsdIYEhmlUN5twY_q0k-RvQ/edit?usp=sharing)

## Repository Structure
Our repository is structured with the following directories:
 * src: contains code for this project, the key files are as follows:
   * db_client.py: contains database client utilities (including connecting, writing, copying, reading, and disconnecting from      the database)
   * db_init.py: populates the database from raw data sources, generates features, and formats data for analysis
   * ml_utils.py: contains machine learning utilities and functions to run analysis
 * data_resources: contains data README file and detailed methodology report from The Eviction Lab
 * resources:
 * tests: contains tests 

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

## Feature Generation
* Run the Census_Data_Cleaning ipython notebook in the data directory to clean the raw census data for import
* Run db_init.py to populate the database with the raw data and generate features
* NOTE: the raw data files should be in the 'src/data/raw' directory on your local machine

## Analysis
* Run ml_utils.py to perform our three phases of analysis and outputs a .csv file with the results from each phase
* Note: for phase 3, the script also produces feature importance .csv files, precision recall graph .png files, and decision     tree .png files. You will need to create 'src/results/csv' and 'src/results/images' directories on your local machine to       capture this output.
* Run the Phase I Analysis, Phase II Analysis, and Phase III Analysis ipython notebooks to replicate the model evaluation and feature importance analysis conducted by the authors at each stage

## Bias Assessment
* To assess our models for bias in Phase III, the authors used the Data Science for Social Good Aequitas Bias Assessment and Audit Toolkig   (http://aequitas.dssg.io/). To replicate this analysis, visit the url, click "get started" and upload the bias csv files output to the 'src/results/csv' repository. 


