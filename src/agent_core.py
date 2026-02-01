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
    """Core agent with ReAct-style reasoning loop"""
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514", max_iterations: int = 3):
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

        # Extract content within <plan> tags first
        plan_match = re.search(r"<plan>(.*?)</plan>", text, re.DOTALL)
        if plan_match:
            text = plan_match.group(1)

        # Parse with error handling
        objective_match = re.search(r"objective:\s*(.*)", text, re.IGNORECASE)
        steps_match = re.search(r"steps:(.*?)next_step:", text, re.DOTALL | re.IGNORECASE)
        next_step_match = re.search(r"next_step:\s*(.*)", text, re.IGNORECASE)

        if not objective_match:
            raise RuntimeError(f"Planner response missing 'objective:' field. Raw response:\n{response.content[0].text}")

        if not steps_match:
            raise RuntimeError(f"Planner response missing 'steps:' field. Raw response:\n{response.content[0].text}")

        if not next_step_match:
            raise RuntimeError(f"Planner response missing 'next_step:' field. Raw response:\n{response.content[0].text}")

        objective = objective_match.group(1).strip()
        steps_block = steps_match.group(1)
        steps = [s.strip("- ").strip() for s in steps_block.splitlines() if s.strip()]
        next_step = next_step_match.group(1).strip()

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
            print(f"TOOL EXECUTED: {tool_name}({tool_input})")
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
            result = str(self.tools[name]["function"](**params))
            return f"[Tool: {name}] {result}"
        except Exception as e:
            return f"Tool execution error: {e}"

    def _critic_prompt(self) -> str:
        return """You are a critic agent that ONLY evaluates progress. You do NOT solve tasks.

Your role:
- Judge whether the objective has been sufficiently achieved based on the observation
- If the observation contains a reasonable answer or useful information, mark as complete
- If more critical information is clearly needed, suggest ONE specific next step

Rules:
- NEVER answer or solve the user's question yourself
- NEVER provide analysis or explanations outside the format
- ONLY output the critique block, nothing else
- Be pragmatic: if the observation addresses the main question, mark complete=true
- Don't be overly perfectionist - "good enough" is complete

You MUST respond with ONLY this exact format (no other text):

<critique>
complete: [true or false]
reason: [one sentence explaining why complete or not]
next_step: [specific action if incomplete, or "none" if complete]
</critique>"""
    
    def critic(self, objective: str, observation: str) -> Critique:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self._critic_prompt(),
            messages=[
                {"role": "user", "content": f"Objective: {objective}\n\nObservation:\n{observation}"}
            ],
        )

        # Debug: inspect full response structure
        print(f"    DEBUG - stop_reason: {response.stop_reason}")
        print(f"    DEBUG - content length: {len(response.content)}")
        print(f"    DEBUG - content[0] type: {type(response.content[0])}")
        print(f"    DEBUG - content[0]: {response.content[0]}")

        text = response.content[0].text
        print(f"    text: {text} \n================================\n")

        # Extract content within <critique> tags first
        critique_match = re.search(r"<critique>(.*?)</critique>", text, re.DOTALL)
        if critique_match:
            text = critique_match.group(1)

        # Parse with error handling
        complete_match = re.search(r"complete:\s*(.*)", text, re.IGNORECASE)
        reason_match = re.search(r"reason:\s*(.*)", text, re.IGNORECASE)
        next_step_match = re.search(r"next_step:\s*(.*)", text, re.IGNORECASE)

        if not complete_match:
            raise RuntimeError(f"Critic response missing 'complete:' field. Raw response:\n{response.content[0].text}")

        if not reason_match:
            raise RuntimeError(f"Critic response missing 'reason:' field. Raw response:\n{response.content[0].text}")

        complete = "true" in complete_match.group(1).lower()
        reason = reason_match.group(1).strip()

        # Parse next_step, treating "none" or empty as None
        next_step = None
        if next_step_match:
            next_step_raw = next_step_match.group(1).strip().lower()
            if next_step_raw and next_step_raw != "none":
                next_step = next_step_match.group(1).strip()

        return Critique(
            complete=complete,
            reason=reason,
            next_step=next_step,
        )

    def run(self, query: str, verbose: bool = True) -> str:
        plan = self.planner(query)
        last_observation = ""

        if verbose:
            print("\n>>> PLAN")
            print(plan)

        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n{'='*60}")
                print(f"ITERATION {iteration + 1}/{self.max_iterations}")
                print(f"{'='*60}")
                print(f"▶ Executing: {plan.next_step}")

            execution = self.executor(plan.next_step)
            last_observation = execution.observation

            if verbose:
                print("\n>>> OBSERVATION")
                print(execution.observation[:500])

            critique = self.critic(plan.objective, execution.observation)

            if verbose:
                print("\n>>> CRITIQUE")
                print(critique)

            if critique.complete:
                if verbose:
                    print("\n✅ TASK COMPLETE")
                return execution.observation

            # On last iteration, return what we have instead of continuing
            if iteration == self.max_iterations - 1:
                if verbose:
                    print("\n⚠️ MAX ITERATIONS REACHED - returning best result")
                return last_observation

            if not critique.next_step:
                if verbose:
                    print("\n⚠️ No next step provided - returning current result")
                return last_observation

            plan.next_step = critique.next_step

        return last_observation