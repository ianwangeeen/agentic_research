# agent_core.py

import anthropic
import json
import re
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
import os

@dataclass
class Plan:
    objective: str
    steps: List[str]
    next_step: str


@dataclass
class ExecutionResult:
    action: Optional[str]
    action_input: Optional[Dict[str, Any]]
    observation: str


@dataclass
class Critique:
    complete: bool
    reason: str
    next_step: Optional[str]


class Agent:
    load_dotenv()

    """Core agent with ReAct-style reasoning loop"""
    
    def __init__(self, api_key: str = os.getenv("API_KEY"), model: str = "claude-sonnet-4-20250514", max_iterations: int = 10):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_iterations = max_iterations
        self.tools: Dict[str, Dict[str, Any]] = {}
        # self.conversation_history: List[Dict] = []
        # self.max_iterations = 10
        
    def register_tool(
        self,
        name: str,
        description: str,
        func: Callable,
        parameters: Dict[str, Any],
    ):
        self.tools[name] = {
            "function": func,
            "description": description,
            "parameters": parameters,
        }

    def _planner_prompt(self) -> str:
        planner_prompt = """
            You are a planning agent.

            Your job:
            - Understand the user's objective
            - Decompose it into concrete steps
            - Decide the SINGLE next step to execute

            Rules:
            - Do NOT call tools
            - Do NOT solve the task
            - Be concise and explicit

            Output format:

            <plan>
            objective: ...
            steps:
            - step 1
            - step 2
            next_step: ...
            </plan>
            """
        return planner_prompt
    
    def planner(self, query: str) -> Plan:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self._planner_prompt(),
            messages=[{"role": "user", "content": query}],
        )

        text = response.content[0].text

        objective = re.search(r"objective:\s*(.*)", text).group(1)
        steps_block = re.search(r"steps:(.*?)next_step:", text, re.DOTALL).group(1)
        steps = [s.strip("- ").strip() for s in steps_block.splitlines() if s.strip()]
        next_step = re.search(r"next_step:\s*(.*)", text).group(1)

        return Plan(objective=objective, steps=steps, next_step=next_step)

    def _executor_prompt(self) -> str:
        tools_desc = "\n\n".join(
            f"Tool: {name}\nDescription: {info['description']}\nParameters:\n{json.dumps(info['parameters'], indent=2)}"
            for name, info in self.tools.items()
        )
        prompt = f"""
            You are an execution agent.

            You are given ONE task step.
            You may either:
            - Call ONE tool
            - OR return a reasoning-only result

            Rules:
            - Never decide completion
            - Never invent tools
            - Use valid JSON if calling a tool

            Available tools:
            {tools_desc}

            Output format (choose ONE):

            <action>tool_name</action>
            <action_input>{{ JSON }}</action_input>

            OR

            <result>
            text
            </result>
            """
        return prompt
    
    def executor(self, step: str) -> ExecutionResult:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=self._executor_prompt(),
            messages=[{"role": "user", "content": step}],
        )

        text = response.content[0].text

        action_match = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
        input_match = re.search(r"<action_input>(.*?)</action_input>", text, re.DOTALL)
        result_match = re.search(r"<result>(.*?)</result>", text, re.DOTALL)

        if action_match:
            tool_name = action_match.group(1).strip()
            tool_input = json.loads(input_match.group(1).strip())

            observation = self._execute_tool(tool_name, tool_input)
            return ExecutionResult(
                action=tool_name,
                action_input=tool_input,
                observation=observation,
            )

        if result_match:
            return ExecutionResult(
                action=None,
                action_input=None,
                observation=result_match.group(1).strip(),
            )

        raise RuntimeError("Invalid executor output")

    def _execute_tool(self, name: str, params: Dict[str, Any]) -> str:
        if name not in self.tools:
            return f"Error: Tool '{name}' not registered."

        try:
            return str(self.tools[name]["function"](**params))
        except Exception as e:
            return f"Tool execution error: {e}"

    def _critic_prompt(self) -> str:
        return """
            You are a critic agent.

            Your role:
            - Objectively evaluate progress
            - Decide if the task is complete
            - If not complete, suggest the next step

            Rules:
            - Be strict
            - Prefer correctness over optimism

            Output format:

            <critique>
            complete: true/false
            reason: ...
            next_step: ...
            </critique>
            """
    
    def critic(self, objective: str, observation: str) -> Critique:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self._critic_prompt(),
            messages=[
                {"role": "user", "content": f"Objective: {objective}\n\nObservation:\n{observation}"}
            ],
        )

        text = response.content[0].text

        complete = "true" in re.search(r"complete:\s*(.*)", text).group(1).lower()
        reason = re.search(r"reason:\s*(.*)", text).group(1)
        next_step_match = re.search(r"next_step:\s*(.*)", text)

        return Critique(
            complete=complete,
            reason=reason,
            next_step=next_step_match.group(1) if next_step_match else None,
        )

    def run(self, query: str, verbose: bool = True) -> str:
        plan = self.planner(query)

        if verbose:
            print("\n📋 PLAN")
            print(plan)

        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n{'='*60}")
                print(f"ITERATION {iteration + 1}")
                print(f"{'='*60}")
                print(f"▶ Executing: {plan.next_step}")

            execution = self.executor(plan.next_step)

            if verbose:
                print("\n🔍 OBSERVATION")
                print(execution.observation[:500])

            critique = self.critic(plan.objective, execution.observation)

            if verbose:
                print("\n🧠 CRITIQUE")
                print(critique)

            if critique.complete:
                if verbose:
                    print("\n✅ TASK COMPLETE")
                return execution.observation

            if not critique.next_step:
                raise RuntimeError("Critic did not provide next step")

            plan.next_step = critique.next_step

        raise RuntimeError("Maximum iterations reached without completion")