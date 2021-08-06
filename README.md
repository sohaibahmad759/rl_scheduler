# Scheduler for BLIS

## Installation instructions 

1. `conda activate scheduler`

2. `sudo yum install openmpi-devel`

3. `source /etc/profile.d/modules.sh`

4. `module load mpi`

5. `pip install -r requirements`

## Architecture

The scheduler makes decision using reinforcement learning (RL)-based agent that has been trained using a simulation environment. The simulation environment captures the scheduling-related events in BLIS and replays a trace of events to the scheduling agent that is then trained on this trace.

The simulation environment consists of four main files:

1. `simulator.py`: responsible for simulating the BLIS events for the scheduler

2. `scheduling_agent.py`: the core part of the scheduling logic, responsible for learning and then making the scheduling decisions for BLIS

3. `profiler.py`: responsible for communicating the profiled information of various models on the different hardware in BLIS to help the `scheduling_agent` make its scheduling decision

4. `resource_monitor.py`: responsible for simulating the resources within BLIS

