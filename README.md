# Parameter Search with WandB Sweep

Please see other CDCD for more insight as to what this is about.

This branch is for running on a machine with multiple cores to find the best parameters.

To run a sweep with multiple cores you need to run the command: ./run_sweep.sh --num_agents 1 --runs_per_agent 100 --sweep_id YOUR_ID

If you leave sweep id blank it will create a new sweep.

run sweep uses weights and biases sweep function and calls sweep_agent.py which initiate the agents that do train runs.