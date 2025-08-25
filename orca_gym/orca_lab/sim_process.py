import argparse
import asyncio

import orca_gym.scripts.run_sim_loop as sim_loop

async def empty_task():
    pass

async def loop():
    await empty_task()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sim_addr",
        type=str,
        required=True,
        help="Simulation address. For example, 'localhost:50051'.",
    )

    args = parser.parse_args()

    sim_loop.run_simulation(args.sim_addr, "OrcaLab Env", "SimulationLoop")
    
    
    # asyncio.run(loop())
