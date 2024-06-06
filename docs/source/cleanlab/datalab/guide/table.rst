.. tabs::

    .. tab:: Classification task

        .. list-table::
            :widths: 20 10 20 50
            :header-rows: 1

            * - Issue Name
              - Default
              - Column Name
              - Required keyword arguments in :py:meth:`Datalab.find_issues <cleanlab.datalab.datalab.Datalab.find_issues>`
            * - :ref:`label<Label Issue>`
              - Yes
              - 
                :ref:`is_label_issue<\`\`is_label_issue\`\`>`

                :ref:`label_score<\`\`label_score\`\`>` 

                :ref:`given_label<\`\`given_label\`\`>`

                :ref:`predicted_label<\`\`predicted_label\`\`>`
              - `pred_probs` or `features`
            * - :ref:`outlier<Outlier Issues>`
              - Yes
              - 
                :ref:`is_outlier_issue<\`\`is_outlier_issue\`\`>`
                
                :ref:`outlier_score<\`\`outlier_score\`\`>`
              - `pred_probs` or `features` or `knn_graph`
            * - :ref:`near_duplicate<(Near) Duplicate Issue>`
              - Yes
              - 
                :ref:`is_near_duplicate_issue<\`\`is_near_duplicate_issue\`\`>`
                
                :ref:`near_duplicate_score<\`\`near_duplicate_score\`\`>`

                :ref:`near_duplicate_sets<\`\`near_duplicate_sets\`\`>`

                :ref:`distance_to_nearest_neighbor<\`\`distance_to_nearest_neighbor\`\`>`
              - `features` or `knn_graph`
            * - :ref:`non_iid<Non-IID Issue>`
              - Yes
              - 
                :ref:`is_non_iid_issue<\`\`is_non_iid_issue\`\`>`
                
                :ref:`non_iid_score<\`\`non_iid_score\`\`>`
              - `pred_probs` or `features` or `knn_graph`
            * - :ref:`class_imbalance<Class Imbalance Issue>`
              - Yes
              - 
                :ref:`is_class_imbalance_issue<\`\`is_class_imbalance_issue\`\`>`
                
                :ref:`class_imbalance_score<\`\`class_imbalance_score\`\`>`
              - None [#f1]_
            * - :ref:`underperforming_group<Underperforming Group Issue>`
              - Yes
              - 
                :ref:`is_underperforming_group_issue<\`\`is_underperforming_group_issue\`\`>`
                
                :ref:`underperforming_group_score<\`\`underperforming_group_score\`\`>`
              - (`pred_probs`, `features`) or (`pred_probs`, `knn_graph`) or (`pred_probs`, `cluster_ids`) [#f2]_
            * - :ref:`null<Null Issue>`
              - Yes
              - 
                :ref:`is_null_issue<\`\`is_null_issue\`\`>`
                
                :ref:`null_score<\`\`null_score\`\`>`
              - `features`
            * - :ref:`data_valuation<Data Valuation Issue>`
              - No
              - 
                :ref:`is_data_valuation_issue<\`\`is_data_valuation_issue\`\`>`
                
                :ref:`data_valuation_score<\`\`data_valuation_score\`\`>`
              - `knn_graph`

        .. rubric:: Notes

        .. [#f1] Only runs if `label_name` is provided in `Datalab()` constructor
        .. [#f2] `cluster_ids` currently needs to be passed via `issue_types`

    .. tab:: Regression task 

        .. list-table::
            :widths: 20 10 20 50
            :header-rows: 1

            * - Issue Name
              - Default
              - Column Name
              - Required keyword arguments in :py:meth:`Datalab.find_issues <cleanlab.datalab.datalab.Datalab.find_issues>`
            * - :ref:`label<Label Issue>`
              - Yes
              - 
                :ref:`is_label_issue<\`\`is_label_issue\`\`>`

                :ref:`label_score<\`\`label_score\`\`>`

                :ref:`given_label<\`\`given_label\`\`>`

                :ref:`predicted_label<\`\`predicted_label\`\`>`
              - `pred_probs` [#f3]_ or `features` or (`features`, `model`) [#f4]_
            * - :ref:`outlier<Outlier Issues>`
              - Yes
              - 
                :ref:`is_outlier_issue<\`\`is_outlier_issue\`\`>`
                
                :ref:`outlier_score<\`\`outlier_score\`\`>`
              - `features` or `knn_graph`
            * - :ref:`near_duplicate<(Near) Duplicate Issue>`
              - Yes
              - 
                :ref:`is_near_duplicate_issue<\`\`is_near_duplicate_issue\`\`>`
                
                :ref:`near_duplicate_score<\`\`near_duplicate_score\`\`>`
              - `features` or `knn_graph`
            * - :ref:`non_iid<Non-IID Issue>`
              - Yes
              - 
                :ref:`is_non_iid_issue<\`\`is_non_iid_issue\`\`>`
                
                :ref:`non_iid_score<\`\`non_iid_score\`\`>`
              - `features` or `knn_graph`
            * - :ref:`null<Null Issue>`
              - Yes
              - 
                :ref:`is_null_issue<\`\`is_null_issue\`\`>`
                
                :ref:`null_score<\`\`null_score\`\`>`
              - `features`

        .. rubric:: Notes

        .. [#f3] :abbr:`pred_probs (Predicted Probabilities)` gets reinterpreted as a `predictions` argument internally
        .. [#f4] `model` currently needs to be passed as `issue_types = {"label": {"clean_learning_kwargs": {"model": your_regression_model}}}`

    .. tab:: Multilabel task 

        .. list-table::
            :widths: 20 10 20 50
            :header-rows: 1

            * - Issue Name
              - Default
              - Column Name
              - Required keyword arguments in :py:meth:`Datalab.find_issues <cleanlab.datalab.datalab.Datalab.find_issues>`
            * - :ref:`label<Label Issue>`
              - Yes
              - 
                :ref:`is_label_issue<\`\`is_label_issue\`\`>`

                :ref:`label_score<\`\`label_score\`\`>`
                
                :ref:`given_label<\`\`given_label\`\`>`

                :ref:`predicted_label<\`\`predicted_label\`\`>` 
              - `pred_probs` or `features`
            * - :ref:`outlier<Outlier Issues>`
              - Yes
              - 
                :ref:`is_outlier_issue<\`\`is_outlier_issue\`\`>`
                
                :ref:`outlier_score<\`\`outlier_score\`\`>`
              - `features` or `knn_graph`
            * - :ref:`near_duplicate<(Near) Duplicate Issue>`
              - Yes
              - 
                :ref:`is_near_duplicate_issue<\`\`is_near_duplicate_issue\`\`>`
                
                :ref:`near_duplicate_score<\`\`near_duplicate_score\`\`>`
              - `features` or `knn_graph`
            * - :ref:`non_iid<Non-IID Issue>`
              - Yes
              - 
                :ref:`is_non_iid_issue<\`\`is_non_iid_issue\`\`>`
                
                :ref:`non_iid_score<\`\`non_iid_score\`\`>`
              - `features` or `knn_graph`
            * - :ref:`null<Null Issue>`
              - Yes
              - 
                :ref:`is_null_issue<\`\`is_null_issue\`\`>`
                
                :ref:`null_score<\`\`null_score\`\`>`
              - `features`