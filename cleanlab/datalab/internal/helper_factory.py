from cleanlab.datalab.internal.adapter.imagelab import (
    ImagelabIssueFinderAdapter,
    ImagelabReporterAdapter,
)
from cleanlab.datalab.internal.issue_finder import IssueFinder
from cleanlab.datalab.internal.report import Reporter


def issue_finder_factory(imagelab):
    if imagelab:
        return ImagelabIssueFinderAdapter
    else:
        return IssueFinder


def report_factory(imagelab):
    if imagelab:
        return ImagelabReporterAdapter
    else:
        return Reporter
