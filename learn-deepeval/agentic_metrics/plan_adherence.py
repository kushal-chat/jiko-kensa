from somewhere import llm
from deepeval.tracing import observe, update_current_trace
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import PlanAdherenceMetric
from deepeval.test_case import ToolCall


@observe
def tool_call(input):
    ...
    return [ToolCall(name="CheckWhether")]

@observe
def agent(input):
    tools = tool_call(input)
    output = llm(input, tools)
    update_current_trace(
        input=input,
        output=output,
        tools_called=tools
    )
    return output


# Create dataset
dataset = EvaluationDataset(goldens=[Golden(input="What's the weather like in SF?")])

# Initialize metric
metric = PlanAdherenceMetric(threshold=0.7, model="gpt-4o")

# Loop through dataset
for golden in dataset.evals_iterator(metrics=[metric]):
    agent(golden.input)