
from typing import List

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


class TestResult(BaseModel):
    test_name: str = Field(description="Name of the test")
    status: str = Field(
        description="passed, failed, or skipped"
    )
    details: str = Field(
        description="Test output or failure reason"
    )


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
        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
        )

        self.structured_llm = self.llm.with_structured_output(
            TesterOutput
        )

    async def analyze_results(
        self,
        task: str,
        implementation_summary: str,
        test_output: str,
        exit_code: int
    ) -> TesterOutput:

        prompt = f"""
        You are the Tester Agent of CodeCrew.

        Your responsibility is to analyze test execution results.

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
        3. Never invent test results.
        4. Identify errors and failure reasons.
        5. If execution failed before tests could run,
           report needs_attention.
        6. If the exit code is non-zero, do not report
           overall_status as passed.
        7. Suggest issues that the Developer Agent should fix.
        8. Do not claim that requirements are satisfied
           without supporting test evidence.

        Return structured test results.
        """

        result = await self.structured_llm.ainvoke(prompt)

        return result
