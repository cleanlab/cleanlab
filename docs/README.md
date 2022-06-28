# Looking for `v1.0` of the `cleanlab` docs?

Please refer to the [`v1.0.1` documentation](https://docs.cleanlab.ai/v1.0.1/);
the code for `v1.0` is identical to the code for `v1.0.1`.

# Looking for rendered docs?

See <https://docs.cleanlab.ai/> if you want to browse the documentation (including for past versions).

# CI/CD for `cleanlab` docs

In the `cleanlab` repository, we've configured GitHub Actions to perform the following automatically:

1. When a commit is pushed to the `master` branch, a new version of the `master` docs will be built and deployed to the `cleanlab-docs` repository.

2. When a release is published, a new version of the docs with the corresponding release tag will be built and deployed as a new folder in the `cleanlab-docs` repository. Redirection to the `stable` version of the docs will be changed to this newly released one, accessible via a link on the docs' site sidebar. All the older versions will remain available in the `cleanlab-docs` repo, accessible by manually entering the subdirectory in the URL.

3. When a user manually runs the workflow, one of the above will happen depending on the user's selection to run from a `branch` or `tag`.

If you'd like to build our docs locally or remotely yourself, or want to know more about the steps taken in the GitHub Pages workflow, read on!


# Build the `cleanlab` docs **locally**

1. [Fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo#forking-a-repository) and [clone](https://docs.github.com/en/get-started/quickstart/fork-a-repo#cloning-your-forked-repository) the `cleanlab` repository.

2. Install the required packages to build the docs:

```
pip install -r docs/requirements.txt
```

3. Install [Pandoc](https://pandoc.org/installing.html).

4. If you don't already have it, install [wget](https://www.gnu.org/software/wget/). This can be done with `brew` on macOS: `brew install wget`

5. **[Optional]** [Create a new branch](https://www.atlassian.com/git/tutorials/using-branches), make your code changes, and then `git commit` them. **ONLY COMMITTED CHANGES WILL BE REFLECTED IN THE DOCS BUILD.**

6. Build the docs with [`sphinx-multiversion`](https://holzhaus.github.io/sphinx-multiversion):

   * If you're building from a **branch** (usually the `master` branch):

   ```
   sphinx-multiversion docs/source cleanlab-docs -D smv_branch_whitelist=YOUR_BRANCH_NAME -D smv_tag_whitelist=None
   ```

   * If you're building from a **tag** (usually the tag of the stable release):

   ```
   sphinx-multiversion docs/source cleanlab-docs -D smv_branch_whitelist=None -D smv_tag_whitelist=YOUR_TAG_NAME
   ```

   Note: To also build docs for another branch or tag, run the above command again changing only the `YOUR_BRANCH_NAME` or `YOUR_TAG_NAME` placeholder.

   **Fast build**: Executing the Jupyter Notebooks (i.e., the `.ipynb` files) that make up some portion of the docs, such as the tutorials, takes a long time. If you want to skip rendering these, set the environment variable `SKIP_NOTEBOOKS=1`. You can either set this using `export SKIP_NOTEBOOKS=1` or do this inline with `SKIP_NOTEBOOKS=1 sphinx-multiversion ...`.

   While building the docs, your terminal might output:
   * `unknown config value 'smv_branch_whitelist' in override, ignoring`, and
   * `unknown config value 'smv_tag_whitelist' in override, ignoring`.

   This is because the `smv_branch_whitelist` and `smv_tag_whitelist` config values are only used by `sphinx-multiversion`, but may also be checked by `sphinx` or other extensions that do not use them. Hence, these can be safely ignored as long as the docs are built correctly.

7. **[Optional]** To show dynamic versioning and version warning banners:

   * Copy the `docs/_templates/versioning.js` file to the `cleanlab-docs/` directory.

   * In the copied `versioning.js` file:

      * find `placeholder_version_number` and replace it with the latest release tag name, and

      * find `placeholder_commit_hash` and replace it with the `master` branch commit hash.

8. **[Optional]** To redirect site visits from `/` or `/stable` to the stable version of the docs:

   * Create a copy of the `docs/_templates/redirect-to-stable.html` file and rename it as `index.html`.

   * In this `index.html` file, find `stable_url` and replace it with `/cleanlab-docs/YOUR_LATEST_RELEASE_TAG_NAME/index.html`.

   * Copy this `index.html` to:

      * `cleanlab-docs/`, and

      * `cleanlab-docs/stable/`.

9. The docs for each branch and/or tag can be found in the `cleanlab-docs/` directory, open any of the `index.html` in your browser to view the docs:

```
cleanlab-docs
|   index.html (redirects to stable release of the docs)
|   versioning.js (for dynamic versioning and version warning banner)
|
└───YOUR_BRANCH_NAME (e.g. master)
│       index.html
│       ...
│
└───YOUR_TAG_NAME_1 (e.g. your stable release tag name)
│       index.html
│       ...
│
└───YOUR_TAG_NAME_2 (e.g. an old release tag name)
│       index.html
│       ...
│
└───stable
│       index.html (redirects to stable release of the docs)
│
└───...
```

# Build the `cleanlab` docs **remotely** on GitHub

1. [Fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo#forking-a-repository) the `cleanlab` repository.

2. [Create a new repository](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site#creating-a-repository-for-your-site) named `cleanlab-docs` and a new branch named `master`.

3. In the `cleanlab-docs` repo, [configure GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site#creating-a-repository-for-your-site); under the **Source** section, select the `master` branch and `/(root)` folder. Take note of the URL where your site is published.

4. [Generate SSH deploy key](https://github.com/peaceiris/actions-gh-pages#user-content-️-create-ssh-deploy-key) and add them to your repos as such:

   * In the `cleanlab-docs` repo, go to **Settings > Deploy Keys > Add deploy key** and add your **public key** with the **Allow write access**
   * In the `cleanlab` repo, go to **Settings > Secrets > New repository secrets** and add your **private key** named `ACTIONS_DEPLOY_KEY`

5. In the `cleanlab` repo, check that you have the **GitHub Pages** workflow under the repo's **Actions** tab. This should be created automatically from `.github\workflows\gh-pages.yaml`. This workflow can be activated by any of the 3 triggers below:

   * A push to the `master` branch in the `cleanlab` repo.
   * Publish of a new release in the `cleanlab` repo.
   * Manually run from the **Run workflow** option and select either the `master` branch or one of the release tag.

6. Activate the workflow with any of the 3 triggers listed above and wait for it to complete.

7. Navigate to the URL where your GitHub Pages site is published in step 3. The default URL should have the format *https://repository_owner.github.io/cleanlab-docs/*.

# Manually adding build artifacts to the `cleanlab-docs` repo

GitHub Actions automatically builds and deploys the docs' build artifacts when [triggered](#cicd-for-cleanlab-docs). If you delete and recreate a release tag, the docs for this tag will be rebuilt and redeployed, hence overwriting the existing artifacts with the new ones.

On rare occasions, you may want to update the docs without deleting and recreating the release tag, for example, when you want to fix a typo in the docs, but you've already deployed your tag to PyPI or Conda. This can be done by manually adding specific docs' build artifacts to the `cleanlab/cleanlab-docs` repo. These steps are for users who have `push` permission to `cleanlab/cleanlab` and `cleanlab/cleanlab-docs` repo.

1. If you haven't already done so, [clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) the `cleanlab/cleanlab` repo.

2. [Create and checkout a new branch](https://git-scm.com/docs/git-checkout).

3. Make the necessary code changes.

4. Perform [git add](https://git-scm.com/docs/git-add) and [git commit](https://git-scm.com/docs/git-commit) for the changes.

5. [git push](https://git-scm.com/docs/git-push) to the `cleanlab/cleanlab` repo. As this is pushed from a non-`master` branch,  GitHub Actions will only build but not deploy the docs' build artifacts.

6. Navigate to [github.com/cleanlab/cleanlab](https://www.github.com/cleanlab/cleanlab) in your browser, select the "Actions" tab, under "Workflow", click "GitHub Pages", then select the workflow that was triggered by the previous step.

7. Ensure that the workflow has completed running.

8. Scroll to the bottom of the page, under "Artifacts", click "docs-html" to download the docs' build artifacts.

9. Unzip "docs-html.zip" and open the "docs-html" folder.

10. Identify the files you would like to replace, i.e., the corresponding files creating the pages on [docs.cleanlab.ai](https://docs.cleanlab.ai).

11. Replace these files in [github.com/cleanlab/cleanlab-docs](https://www.github.com/cleanlab/cleanlab-docs) by uploading the new ones to the corresponding version folder in the `master` branch of the `cleanlab/cleanlab-docs` repo.

> :warning: Any build artifacts manually added to `cleanlab/cleanlab-docs` that do not live in the `master` branch of the `cleanlab/cleanlab` repo will be lost in future versions of cleanlab docs. So any edit made in the v2.0.0 docs which you also want to have in the v2.0.1, v2.0.2, etc. docs needs to be introduced as a PR to the `cleanlab/cleanlab` repo as well.

# Behind-the-scenes of the GitHub Pages workflow

We've configured GitHub Actions to run the GitHub Pages workflow (gh-pages.yaml) to build and deploy our docs' static files. Here's a breakdown of what this workflow does in the background:

## Spin up and configure the CI/CD server

1. Spin up a Ubuntu server.

2. Install Pandoc, a document converter required by `nbsphinx` to generate static sites from notebooks (`.ipynb`).

3. Check-out the `cleanlab` repository.

4. Setup Python and cache dependencies.

5. Install dependencies for the docs from `docs/requirements.txt`.

## Build the docs' static site files

6. Run Sphinx with the `sphinx-multiversion` wrapper to build the doc's static site files. These files will be outputted to the `cleanlab-docs/` directory.

## Generate the `versioning.js` file used to store the latest release tag name and commit hash

7. Get the latest release tag name and insert it in the `versioning.js` file. The `index.html` of each doc version will read this as a variable and display it beside the **stable** hyperlink.

8. Insert the latest commit hash in the `versioning.js` file. The `index.html` of each doc version will read this as a variable and display it beside the **developer** hyperlink.

9. Copy the `versioning.js` file to the `cleanlab-docs/` folder.

## If the workflow is **triggered by a new release**, generate the redirecting HTML which redirects site visits to the stable version

10. Insert the relative path to the stable docs in the `redirect-to-stable.html` file AKA the *redirecting HTML*.

11. Create a copy of the `redirect-to-stable.html` file to `cleanlab-docs/index.html` and `cleanlab-docs/index.html`.

## Deploy the static files

12. Deploy `cleanlab-docs/` folder to the `cleanlab/cleanlab-docs` repo's `master branch`.
