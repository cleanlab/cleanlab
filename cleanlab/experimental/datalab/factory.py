from __future__ import annotations

from typing import List, Type

from cleanlab.experimental.datalab.issue_manager import (
    HealthIssueManager,
    IssueManager,
    LabelIssueManager,
)


# Construct concrete issue manager with a from_str method
class _IssueManagerFactory:
    """Factory class for constructing concrete issue managers."""

    types = {
        "health": HealthIssueManager,
        "label": LabelIssueManager,
    }

    @classmethod
    def from_str(cls, issue_type: str) -> Type[IssueManager]:
        """Constructs a concrete issue manager class from a string."""
        if isinstance(issue_type, list):
            raise ValueError(
                "issue_type must be a string, not a list. Try using from_list instead."
            )
        if issue_type not in cls.types:
            raise ValueError(f"Invalid issue type: {issue_type}")
        return cls.types[issue_type]

    @classmethod
    def from_list(cls, issue_types: List[str]) -> List[Type[IssueManager]]:
        """Constructs a list of concrete issue manager classes from a list of strings."""
        return [cls.from_str(issue_type) for issue_type in issue_types]
