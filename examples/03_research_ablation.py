"""Running a controlled multi-tangent ablation benchmark across baseline modes."""

import asyncio

from chatree import BenchmarkRunner, default_multi_tangent_scenario


async def main() -> None:
    runner = BenchmarkRunner(model="mock/chatree-demo")
    report = await runner.run_suite(default_multi_tangent_scenario())
    print(report.to_markdown_table())


if __name__ == "__main__":
    asyncio.run(main())
