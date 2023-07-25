from cleanlab.datalab.adapter.imagelab import (
    ImagelabIssueFinderAdapter,
    ImagelabDataIssuesAdapter,
    ImagelabReporterAdapter,
)
from cleanlab.datalab.data_issues import DataIssues
from cleanlab.datalab.issue_finder import IssueFinder
from cleanlab.datalab.report import Reporter


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
