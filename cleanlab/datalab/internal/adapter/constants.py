SPURIOUS_CORRELATION_ISSUE = {
    "spurious_correlations": {
        "threshold": 0.3,
    },
}  # Default issue type for spurious correlation in Datalab

DEFAULT_CLEANVISION_ISSUES = {
    "dark": {},
    "light": {},
    "low_information": {
        "threshold": 0.15,
    },
    "odd_aspect_ratio": {},
    "odd_size": {},
    "grayscale": {},
    "blurry": {},
}  # list of CleanVision issue types considered by default in Datalab may differ from CleanVision package default

IMAGELAB_ISSUES_MAX_PREVALENCE = 0.1
