from app.agents.agent import Agent
from app.data.trajectory import Trajectory
from app.evaluation.benchmark import BenchmarkCase
from app.evaluation.metrics import task_success


async def generate_trajectory(agent: Agent, case: BenchmarkCase, temperature: float = 0) -> Trajectory:
    state = await agent.run(case.task, temperature=temperature)
    success = await task_success(state, case.expected_answer)
    return Trajectory.from_state(state, success)


async def generate_trajectories(
    agent: Agent, cases: list[BenchmarkCase], temperature: float = 0, runs_per_case: int = 1
) -> list[Trajectory]:
    return [
        await generate_trajectory(agent, case, temperature=temperature)
        for case in cases
        for _ in range(runs_per_case)
    ]
