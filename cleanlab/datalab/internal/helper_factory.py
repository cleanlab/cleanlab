from cleanlab.datalab.internal.adapter.imagelab import (
    ImagelabIssueFinderAdapter,
    ImagelabDataIssuesAdapter,
    ImagelabReporterAdapter,
)
from cleanlab.datalab.internal.data_issues import DataIssues
from cleanlab.datalab.internal.issue_finder import IssueFinder
from cleanlab.datalab.internal.report import Reporter


def issue_finder_factory(imagelab):
    if imagelab:
        return ImagelabIssueFinderAdapter
    else:
        return IssueFinder


def data_issues_factory(imagelab):
    if imagelab:
        return ImagelabDataIssuesAdapter
    else:
        return DataIssues


def report_factory(imagelab):
    if imagelab:
        return ImagelabReporterAdapter
    else:
        return Reporter
