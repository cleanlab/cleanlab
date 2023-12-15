Generating Cluster IDs
======================

The underperforming group issue manager provides the option for passing pre-computed
cluster IDs to `find_issues`. These cluster IDs can be obtained by clustering
the features using algorithms such as K-Means, DBSCAN, HDBSCAN etc. Note that

* K-Means requires specifying the number of clusters explicitly.
* DBSCAN is sensitive to the choice of `eps` (radius) and `min_samples` (minimum samples for each cluster).


Example:

.. code-block:: python

    import datalab
    from sklearn.cluster import KMeans
    features, labels = your_data() # Get features and labels
    pred_probs = get_pred_probs() # Get prediction probabilities for all samples
    # Group features into 8 clusters
    clusterer = KMeans(n_clusters=5)
    clusterer.fit(features)
    cluster_ids = clusterer.labels_
    lab = Datalab(data={"features": features, "y": labels}, label_name="y")
    issue_types = {"underperforming_group": {"cluster_ids": cluster_ids}}
    lab.find_issues(features=features, pred_probs=pred_probs, issue_types=issue_types)



