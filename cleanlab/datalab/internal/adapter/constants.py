DEFAULT_CLEANVISION_ISSUES = {
    "dark": {"threshold": 0.32},
    "light": {"threshold": 0.05},
    "low_information": {
        "threshold": 0.15,
        "normalizing_factor": 0.1,
    },
    "odd_aspect_ratio": {"threshold": 0.35},
    "odd_size": {"threshold": 10.0},
    "grayscale": {},
    "blurry": {
        "threshold": 0.29,
        "normalizing_factor": 0.01,
        "color_threshold": 0.18,
    },
}  # list of CleanVision issue types considered by default in Datalab may differ from CleanVision package default

IMAGELAB_ISSUES_MAX_PREVALENCE = 0.1
