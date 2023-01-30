# Copyright (C) 2017-2023  Cleanlab Inc.
# This file is part of cleanlab.
#
# cleanlab is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cleanlab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with cleanlab.  If not, see <https://www.gnu.org/licenses/>.
"""The factory module provides a factory class for constructing concrete issue managers
and a decorator for registering new issue managers.

This module provides the :py:meth:`register` decorator for users to register new subclasses
of :py:class:`IssueManager <cleanlab.experimental.datalab.issue_manager.IssueManager>` in the registry.


Note
----

The :class:`REGISTRY` variable is used by the factory class to keep track of registered issue managers.
The factory class is used as an implementation detail by :py:class:`Datalab <cleanlab.experimental.datalab.datalab.Datalab>`,
which provides a simplified API for constructing concrete issue managers.
:py:class:`Datalab <cleanlab.experimental.datalab.datalab.Datalab>` is intended to be used by users
and provides detailed documentation on how to use the API.

Warning
-------
Neither the ``REGISTRY`` variable nor the factory class should be used directly by users.
"""
from __future__ import annotations

from typing import Dict, List, Type

from cleanlab.experimental.datalab.issue_manager import (
    IssueManager,
    LabelIssueManager,
    NearDuplicateIssueManager,
    OutOfDistributionIssueManager,
)


REGISTRY: Dict[str, Type[IssueManager]] = {
    "outlier": OutOfDistributionIssueManager,
    "label": LabelIssueManager,
    "near_duplicate": NearDuplicateIssueManager,
}
"""Registry of issue managers that can be constructed from a string 
and used in the Datalab class.

Currently, the following issue managers are registered by default:

- ``outlier``: :py:class:`OutOfDistributionIssueManager <cleanlab.experimental.datalab.issue_manager.OutOfDistributionIssueManager>`
- ``label``: :py:class:`LabelIssueManager <cleanlab.experimental.datalab.issue_manager.LabelIssueManager>`
- ``near_duplicate``: :py:class:`NearDuplicateIssueManager <cleanlab.experimental.datalab.issue_manager.NearDuplicateIssueManager>`

Warning
-------
This variable should not be used directly by users.
"""


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
        return [cls.from_str(issue_type) for issue_type in issue_types]


def register(cls: Type[IssueManager]) -> Type[IssueManager]:
    """Registers the issue manager factory.

    Parameters
    ----------
    cls :
        A subclass of :py:class:`IssueManager <cleanlab.experimental.datalab.issue_manager.IssueManager>`.

    Returns
    -------
    cls :
        The same class that was passed in.

    Example
    -------

    When defining a new subclass of IssueManager, you can register it like so:

    .. code-block:: python

        from cleanlab.experimental.datalab.issue_manager import IssueManager
        from cleanlab.experimental.datalab.factory import register

        @register
        class MyIssueManager(IssueManager):
            issue_name: str = "my_issue"
            def find_issues(self, **kwargs):
                # Some logic to find issues
                pass
    """
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
