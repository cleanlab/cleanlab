.. tip::

        This type of issue has the issue name `"{{issue_name}}"`.

        Run a check for this particular kind of issue by calling :py:meth:`Datalab.find_issues() <cleanlab.datalab.datalab.Datalab.find_issues>` like so:

        .. code-block:: python

            # `lab` is a Datalab instance
            lab.find_issues(..., issue_types = {"{{issue_name}}": {}})
