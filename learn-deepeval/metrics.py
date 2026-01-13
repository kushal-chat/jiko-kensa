
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import ToolCorrectnessMetric

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    # Replace this with the tools that was actually used by your LLM agent
    tools_called=[ToolCall(name="WebSearch"), ToolCall(name="ToolQuery")],
    expected_tools=[ToolCall(name="WebSearch")],
)
metric = ToolCorrectnessMetric()

# To run metric as a standalone
# metric.measure(test_case)
# print(metric.score, metric.reason)

evaluate(test_cases=[test_case], metrics=[metric])