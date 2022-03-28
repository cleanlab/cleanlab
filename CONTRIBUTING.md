# Contributing to cleanlab

All kinds of contributions to cleanlab are greatly appreciated. If you're not looking to write code, submitting a [feature request](#feature-requests) or
[bug report](#bug-reports) is a great way to contribute. If you want to get
your hands dirty, you can submit [Pull Requests](#pull-requests), either working on your
own ideas or addressing [existing issues][issues].

If you are unsure or confused about anything, please go ahead and submit your
issue or pull request anyways! We appreciate all contributions, and we'll do
our best to incorporate your feedback or code into cleanlab.

## Feature Requests

Do you have an idea for an awesome new feature for cleanab? Let us know by
[submitting a feature request][issue].

If you are inclined to do so, you're welcome to [fork][fork] cleanlab, work on
implementing the feature yourself, and submit a patch. In this case, it's
*highly recommended* that you first [open an issue][issue] describing your
enhancement to get early feedback on the new feature that you are implementing.
This will help avoid wasted efforts and ensure that your work is incorporated
into the cleanlab code base.

## Bug Reports

Did something go wrong with cleanlab? Sorry about that! Bug reports are greatly
appreciated!

When you [submit a bug report][issue], please include as much context as you
can. This includes information like Python version, cleanlab version, an error
message or stack trace, and detailed steps to reproduce the bug (if possible, including toy data that reproduces the error). The more information you can include, the better.

## Pull Requests

Want to write code to help improve cleanlab? Awesome!

If there are [open issues][issues], you're more than welcome to work on those (a good place to start is those tagged "good first issue"). If you have your own ideas, that's great too! In that case, before working on substantial changes to the code base, it is *highly recommended* that you first
[open an issue][issue] describing what you intend to work on.

To contribute your code to the library, you'll want to create a new [Pull Request][pr]. 

Any changes to the code base should try to follow the style and coding conventions used in the rest of the project. Cleanlab follows the [Black] style. Before you submit your pull request, install Black with `pip install black` and format the code with `black`. New functions should be well-documented with accompanying unit tests that run quickly.

Once you have finalized your edits to the cleanlab code, make sure the unit tests pass by executing: `pytest` from the root cleanlab/ directory.

You can optionally [build the docs from your local cleanlab version][instructions] to check that the documentation for your new functions is formatted correctly and the existing functions' documentation remains valid. And if you made major changes, you may wish to check that the [examples] still work with the new code.

---

If you have any questions about contributing to cleanlab, feel free to
[ask][discussions]!

[issue]: https://github.com/cleanlab/cleanlab/issues/new
[issues]: https://github.com/cleanlab/cleanlab/issues
[fork]: https://github.com/cleanlab/cleanlab/fork
[pr]: https://github.com/cleanlab/cleanlab/pulls
[discussions]: https://github.com/cleanlab/cleanlab/discussions
[examples]: https://github.com/cleanlab/examples
[instructions]: https://github.com/cleanlab/cleanlab/blob/master/docs/README.md#build-the-cleanlab-docs-locally
[Black]: https://github.com/psf/black
