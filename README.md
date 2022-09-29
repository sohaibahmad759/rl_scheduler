# Scheduler for BLIS

## Installation instructions 

1. `conda create --name scheduler python=3.7.7`

2. `conda activate scheduler`

3. `pip install -r requirements.txt`

## Architecture

The scheduler makes decisions using a reinforcement learning (RL)-based agent that has been trained using a simulation environment. The simulation environment captures the scheduling-related events in BLIS and replays a trace of events to the scheduling agent that is then trained on this trace.

The RL learning environment consists of four main files:

1. `train.py`: this is the file that is called to train the RL agent. After training, it saves the trained model under `saved_models/`

2. `test.py`: loads a model from `saved_models/` and tests it on a trace. The model architecture that is loaded must be the same as used in `train.py`

3. `scheduling_env.py`: the core part of the scheduling logic, responsible for learning and then making the scheduling decisions for BLIS. Interacts with `simulator.py` to make 

4. `simulator.py`: responsible for simulating the BLIS events for the scheduler. Uses the `Executor` class from `executor.py`

5. `executor.py`: contains the code to simulate BLIS executors. Also contains the logic for the local scheduler. Uses the `Predictor` class from `predictor.py`

6. `predictor.py`: contains the code to simulate BLIS predictors

6. `generate_synthetic_trace.py`: generates synthetic traces using a Poisson arrival process

# List of To-dos:
This is a list of general to-dos. Apart from this, every code file has to-dos at specific points for things that might need to be changed.

1. Make `train.py`, `test.py` and `generate_synthetic_trace.py` parameterized with argparse

2. Sparse ISIs when max limit on number of ISIs is large


# Reproducing results from SoCC 2022 submission:

To activate conda environment:

`conda activate scheduler`

To run ILP with Earliest Finish Time (EFT-FIFO):

`python3 test.py -ma 7 -js 4 -r -f 10 -p traces/azure --alpha 0.1 --beta 0.8`

Various scripts used for plotting results:

`python3 plot_vs_infaas.py`

`python3 cdf_response_times_per_model.py`

`python3 per_model_throughput.py`
