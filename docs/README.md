# Build the `cleanlab` docs **locally**

1. [Fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo#forking-a-repository) and [clone](https://docs.github.com/en/get-started/quickstart/fork-a-repo#cloning-your-forked-repository) the `cleanlab` repository.

2. Install the doc's required packages:

```
pip install -r docs/requirements.txt
```

3. Build the docs for all branches and tags with [`sphinx-multiversion`](https://holzhaus.github.io/sphinx-multiversion):

```
sphinx-multiversion docs/source docs/build
```

4. The docs for each branch and tag can be found in the `docs/build` directory, open the `index.html` in your browser to view the docs:

```
docs
│
└───build
│   │
│   └───branch_1
│   │       index.html
│   │       ...
│   │
│   └───tag_1
│   │       index.html
│   │       ...
│   │
│   └───tag_2
│   │       index.html
│   │       ...
│   │
│   └───...
│
└───...
    │   ...
```

# Build the `cleanlab` docs **remotely** on GitHub

1. [Fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo#forking-a-repository) the `cleanlab` repository.

2. [Create a new repository](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site#creating-a-repository-for-your-site) named `cleanlab-docs` and a new branch named `master`.

3. In the `cleanlab-docs` repo, [configure GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site#creating-a-repository-for-your-site); under the **Source** section, select the `master` branch and `/(root)` folder. Take note of the URL where your site is published.

4. [Generate SSH deploy key](https://github.com/peaceiris/actions-gh-pages#%EF%B8%8F-create-ssh-deploy-key) and add them to your repos as such:

   - In the `cleanlab-docs` repo, go to **Settings > Deploy Keys > Add deploy key** and add your **public key** with the **Allow write access**
   - In the `cleanlab` repo, go to **Settings > Secrets > New repository secrets** and add your **private key** named `ACTIONS_DEPLOY_KEY`

5. In the `cleanlab` repo, check that you have the **GitHub Pages** workflow under the repo's **Actions** tab. This should be created automatically from `.github\workflows\gh-pages.yaml`. This workflow can be activated by any of the 3 triggers below:

   1. Pushes to the `master` branch in the `cleanlab` repo.
   2. Publish of a new release in the `cleanlab` repo.
   3. Manually run from the **Run workflow** option and selecting either the `master` branch or one of the release tag.

6. Activate the workflow with any of the 3 triggers listed above and wait for it to complete.

7. Navigate to the URL where your GitHub Pages site is published in step 3. The default URL should have the format *https://repository_owner.github.io/cleanlab-docs/*

# Behind-the-scenes of the GitHub Pages workflow

We've configured GitHub Actions to run the GitHub Pages workflow (gh-pages.yaml) to build and deploy our docs' static files. Here's a breakdown of what this workflow does in the background:

## Spin up and configure the CI/CD server

1. Spin up a Ubuntu server.

2. Install Pandoc, a document converter required by `nbsphinx` to generate static sites from notebooks (`.ipynb`).

3. Check-out the `cleanlab` repository.

4. Setup Python and cache dependencies.

5. Install dependencies for the docs from `docs/requirements.txt`.

## Build the docs' static site files

6. Run Sphinx with `sphinx-multiversion` wrapper to build the docs' static site files. These files will be outputted to the `docs/build` directory.

## Generate the `versioning.js` file used to store the latest release id and commit hash

7. Get the latest release id and insert it in the `versioning.js` file. The `index.html` of each doc version will read this as a variable and display it beside the **stable** hyperlink.

8. Insert the latest commit hash in the `versioning.js` file. The `index.html` of each doc version will read this as a variable and display it beside the **developer** hyperlink.

9. Copy the `versioning.js` file to the `docs/build` folder. 

## If the workflow is **triggered by a new release**, generate the redirecting HTML which redirects site visits to the latest release

10. Insert the repository owner name in the `redirect-to-stable.html` file AKA the *redirecting HTML*.

11. Create a copy of the `redirect-to-stable.html` file to `docs/build` then insert the stable release docs's `index.html` file path prefixed with ``.``

12. Create a copy of the `redirect-to-stable.html` file to `docs/build/stable` then insert the stable release docs's `index.html` file path prefixed with ``..``

## Deploy the static files

13. Deploy `docs/build` folder to the `cleanlab/cleanlab-docs` repo's `master branch`