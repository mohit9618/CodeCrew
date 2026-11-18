
import json
import re
from typing import List

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from backend.app.llm import get_groq_llm
from backend.app.utils.text import clamp_text

load_dotenv()


class TestResult(BaseModel):
    test_name: str = Field(description="Name of the test")
    status: str = Field(description="passed, failed, or skipped")
    details: str = Field(description="Test output or failure reason")


class TesterOutput(BaseModel):
    overall_status: str = Field(
        description="passed, failed, or needs_attention"
    )
    passed_count: int
    failed_count: int
    skipped_count: int
    results: List[TestResult]
    analysis: str = Field(
        description="Analysis of the test execution"
    )
    issues: List[str] = Field(
        default_factory=list,
        description="Issues that need developer attention"
    )


class TesterAgent:

    def __init__(self):
        self.llm = get_groq_llm()

    def extract_json(self, content: str) -> dict:
        content = content.strip()

        content = re.sub(
            r"```(?:json)?\s*",
            "",
            content,
            flags=re.IGNORECASE
        )
        content = content.replace("```", "").strip()

        start = content.find("{")

        if start == -1:
            raise ValueError("No JSON object found in model response")

        decoder = json.JSONDecoder()
        result, _ = decoder.raw_decode(content[start:])

        return result

    async def analyze_results(
        self,
        task: str,
        implementation_summary: str,
        test_output: str,
        exit_code: int
    ) -> TesterOutput:

        # Keep the prompt inside the Groq token budget. pytest prints its
        # failure summary last, so keep the tail of the output.
        task = clamp_text(task, 1_500, "tester task")
        implementation_summary = clamp_text(
            implementation_summary, 1_500, "implementation summary"
        )
        test_output = clamp_text(
            test_output, 8_000, "test output", keep="tail"
        )

        prompt = f"""
You are the Tester Agent of CodeCrew.

Analyze the provided test execution results.

DEVELOPMENT TASK:
{task}

IMPLEMENTATION SUMMARY:
{implementation_summary}

TEST EXECUTION OUTPUT:
{test_output}

EXIT CODE:
{exit_code}

Instructions:
1. Analyze only the provided execution output.
2. Identify passed, failed, and skipped tests.
3. Never invent test names or test results.
4. Identify errors and failure reasons.
5. If execution failed before tests could run, use needs_attention.
6. If the exit code is non-zero, never report passed.
7. Suggest actionable issues for the Developer Agent.
8. Do not claim requirements are satisfied without test evidence.
9. Distinguish test failures from environment or dependency errors.
10. If no tests actually executed, all three counts must be zero.
11. Return only one valid JSON object. Do not include explanations,
    markdown, code fences, or reasoning outside the JSON.

Required JSON structure:
{{
    "overall_status": "passed | failed | needs_attention",
    "passed_count": 0,
    "failed_count": 0,
    "skipped_count": 0,
    "results": [
        {{
            "test_name": "test name",
            "status": "passed | failed | skipped",
            "details": "test details"
        }}
    ],
    "analysis": "analysis of execution",
    "issues": ["actionable issue"]
}}
"""

        try:
            response = await self.llm.ainvoke(prompt)

            content = response.content

            if isinstance(content, list):
                content = "".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict)
                )

            data = self.extract_json(str(content))

            result = TesterOutput.model_validate(data)

            if exit_code != 0 and result.overall_status == "passed":
                result.overall_status = "needs_attention"

            return result

        except Exception as error:
            print(f"TESTER AGENT PARSING ERROR: {error}")

            return TesterOutput(
                overall_status="needs_attention",
                passed_count=0,
                failed_count=0,
                skipped_count=0,
                results=[],
                analysis=(
                    "Tester Agent could not reliably parse the model response. "
                    "The test execution output requires manual review."
                ),
                issues=[
                    f"Tester Agent response parsing failed: {str(error)}",
                    "Review the original test execution output."
                ]
            )
