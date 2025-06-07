from dotenv import load_dotenv
from groq import Groq
import re

load_dotenv()

groq = Groq()


class LLMClassifier:
    """
    A classifier that uses a large language model to categorize log messages.
    """

    def __init__(self):
        self.groq = groq

    def classify_with_llm(self, log_msg, model="deepseek-r1-distill-llama-70b"):
        """
        Classify the log message into a category.

        Args:
            log_msg (str): The log message to classify.
            model (str): The LLM model to use.

        Returns:
            str: The category assigned to the log message.
        """
        prompt = (
            "Classify the log message into one of these categories: "
            "(1) Workflow Error, (2) Deprecation Warning.\n"
            "If you can't figure out a category, use \"Unclassified\".\n"
            "Put the category inside <category> </category> tags.\n"
            f"Log message: {log_msg}"
        )

        chat_completion = self.groq.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.5
        )

        content = chat_completion.choices[0].message.content
        match = re.search(r'<category>(.*)<\/category>', content, flags=re.DOTALL)
        category = "Unclassified"
        if match:
            category = match.group(1)

        return category

if __name__ == "__main__":
    classifier = LLMClassifier()
    print(classifier.classify_with_llm(
        "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active."))
    print(classifier.classify_with_llm(
        "The 'ReportGenerator' module will be retired in version 4.0. Please migrate to the 'AdvancedAnalyticsSuite' by Dec 2025"))
    print(classifier.classify_with_llm("System reboot initiated by user 12345."))