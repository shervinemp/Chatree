"""Run the full 5-family Chatree empirical evaluation suite with leakage, staleness, cost, and latency metrics."""

import asyncio

from chatree import BenchmarkRunner, build_full_eval_suite


async def main() -> None:
    runner = BenchmarkRunner(
        model="mock/chatree-demo",
        judge_model="mock/chatree-demo",
    )
    suite_report = await runner.run_scenarios(build_full_eval_suite(num_distractors=4))
    print(suite_report.to_markdown_table())


if __name__ == "__main__":
    asyncio.run(main())
