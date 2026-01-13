from deepeval import evaluate
from deepeval.metrics import ArgumentCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

metric = ArgumentCorrectnessMetric(
    threshold=0.7,
    model="gpt-4",
    include_reason=True
)
test_case = LLMTestCase(
    input="When did Trump first raise tariffs?",
    actual_output="Trump first raised tariffs in 2018 during the U.S.-China trade war.",
    tools_called=[
        ToolCall(
            name="WebSearch Tool",
            description="Tool to search for information on the web.",
            input={"search_query": "Trump first raised tariffs year"}
        ),
        ToolCall(
            name="History FunFact Tool",
            description="Tool to provide a fun fact about the topic.",
            input={"topic": "Trump tariffs"}
        )
    ]
)

# To run metric as a standalone
# metric.measure(test_case)
# print(metric.score, metric.reason)

evaluate(test_cases=[test_case], metrics=[metric])