from __future__ import annotations

from typing import Dict, List, Type

from cleanlab.experimental.datalab.issue_manager import (
    IssueManager,
    LabelIssueManager,
    OutOfDistributionIssueManager,
    NearDuplicateIssueManager,
)


REGISTRY: Dict[str, Type[IssueManager]] = {
    "outlier": OutOfDistributionIssueManager,
    "label": LabelIssueManager,
    "near_duplicate": NearDuplicateIssueManager,
}
# Construct concrete issue manager with a from_str method
class _IssueManagerFactory:
    """Factory class for constructing concrete issue managers."""

    @classmethod
    def from_str(cls, issue_type: str) -> Type[IssueManager]:
        """Constructs a concrete issue manager class from a string."""
        if isinstance(issue_type, list):
            raise ValueError(
                "issue_type must be a string, not a list. Try using from_list instead."
            )
        if issue_type not in REGISTRY:
            raise ValueError(f"Invalid issue type: {issue_type}")
        return REGISTRY[issue_type]

    @classmethod
    def from_list(cls, issue_types: List[str]) -> List[Type[IssueManager]]:
        """Constructs a list of concrete issue manager classes from a list of strings."""
        return [REGISTRY[issue_type] for issue_type in issue_types]


def register(cls: Type[IssueManager]):
    """Registers the issue manager factory."""
    name: str = str(cls.issue_name)
    if name in REGISTRY:
        # Warn user that they are overwriting an existing issue manager
        print(
            f"Warning: Overwriting existing issue manager {name} with {cls}. "
            "This may cause unexpected behavior."
        )
    if not issubclass(cls, IssueManager):
        raise ValueError(f"Class {cls} must be a subclass of IssueManager")
    REGISTRY[name] = cls
    return cls
