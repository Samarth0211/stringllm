"""Pre-built prompt templates for common LLM tasks."""

from __future__ import annotations

from stringllm.prompts.template import PromptTemplate


class PromptLibrary:
    """Collection of ready-to-use :class:`PromptTemplate` instances.

    All methods are static and return a fresh :class:`PromptTemplate` on each call,
    so callers can safely mutate defaults without affecting other users.
    """

    @staticmethod
    def summarize() -> PromptTemplate:
        """Summarise text into bullet points.

        Variables:
            num_points (default ``"3"``): Number of bullet points.
            text: The text to summarise.
        """
        return PromptTemplate(
            template=(
                "Summarize the following text into {num_points} bullet points:\n\n{text}"
            ),
            defaults={"num_points": "3"},
        )

    @staticmethod
    def analyze_sentiment() -> PromptTemplate:
        """Analyse the sentiment of a piece of text.

        Variables:
            text: The text to analyse.
        """
        return PromptTemplate(
            template=(
                "Analyze the sentiment of the following text. Classify it as "
                "positive, negative, or neutral and provide a brief explanation:\n\n{text}"
            ),
        )

    @staticmethod
    def translate() -> PromptTemplate:
        """Translate text to a target language.

        Variables:
            target_language: The language to translate into.
            text: The text to translate.
        """
        return PromptTemplate(
            template="Translate the following text to {target_language}:\n\n{text}",
        )

    @staticmethod
    def extract_keywords() -> PromptTemplate:
        """Extract top keywords from text.

        Variables:
            num_keywords (default ``"5"``): How many keywords to extract.
            text: The text to process.
        """
        return PromptTemplate(
            template=(
                "Extract the top {num_keywords} keywords from the following text "
                "and return them as a comma-separated list:\n\n{text}"
            ),
            defaults={"num_keywords": "5"},
        )

    @staticmethod
    def rewrite() -> PromptTemplate:
        """Rewrite text in a specified tone.

        Variables:
            tone: The desired tone (e.g. ``"formal"``, ``"casual"``).
            text: The text to rewrite.
        """
        return PromptTemplate(
            template="Rewrite the following text in a {tone} tone:\n\n{text}",
        )

    @staticmethod
    def code_review() -> PromptTemplate:
        """Review code for bugs, improvements, and best practices.

        Variables:
            code: The source code to review.
        """
        return PromptTemplate(
            template=(
                "Review the following code for bugs, potential improvements, and "
                "adherence to best practices. Provide actionable feedback:\n\n{code}"
            ),
        )

    @staticmethod
    def explain_code() -> PromptTemplate:
        """Explain what a piece of code does in plain English.

        Variables:
            code: The source code to explain.
        """
        return PromptTemplate(
            template="Explain what this code does in plain English:\n\n{code}",
        )
