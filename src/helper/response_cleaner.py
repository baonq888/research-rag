import re

class ResponseCleaner:
    @staticmethod
    def strip_think_block(text: str) -> str:
        """Remove <think>, <thinking>, <reasoning> blocks (DeepSeek, Claude, etc.)"""
        return re.sub(r"<(think|thinking|reasoning)>.*?</\1>\s*", "", text, flags=re.DOTALL).strip()
    