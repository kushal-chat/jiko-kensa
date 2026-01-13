from deepeval.tracing import observe, update_current_span
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import AnswerRelevancyMetric

# Classic example

@observe()
def llm_app(input: str):

    @observe(metrics=[AnswerRelevancyMetric()])
    def inner_component():
        update_current_span(test_case=LLMTestCase(input="Why is the blue sky?", actual_output="You mean why is the sky blue?"))
    
    return inner_component()

dataset = EvaluationDataset(goldens=[Golden(input="Test input")])

for golden in dataset.evals_iterator():
    llm_app(golden.input)