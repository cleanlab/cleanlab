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