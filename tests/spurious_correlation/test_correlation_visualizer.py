from unittest.mock import patch

from cleanlab.datalab.internal.adapter.imagelab import CorrelationVisualizer

VIZMANAGER_IMPORT_PATH = "cleanvision.utils.viz_manager.VizManager"


class TestCorrelationVisualizer:
    def test_correlation_visualizer_init(self):
        with patch(VIZMANAGER_IMPORT_PATH) as mock_viz_manager:
            visualizer = CorrelationVisualizer()
            assert visualizer.viz_manager == mock_viz_manager

    def test_visualize(self):
        with patch(VIZMANAGER_IMPORT_PATH) as mock_viz_manager:
            visualizer = CorrelationVisualizer()

            images = ["image1", "image2", "image3"]
            title_info = {"scores": ["score1", "score2", "score3"]}
            ncols = 2
            cell_size = (2, 2)

            visualizer.visualize(images, title_info)

            mock_viz_manager.individual_images.assert_called_once_with(
                images=images,
                title_info=title_info,
                ncols=ncols,
                cell_size=cell_size,
            )

    def test_visualize_custom_params(self):
        with patch(VIZMANAGER_IMPORT_PATH) as mock_viz_manager:
            visualizer = CorrelationVisualizer()

            images = ["image1", "image2", "image3"]
            title_info = {"scores": ["score1", "score2", "score3"]}
            ncols = 3
            cell_size = (3, 3)

            visualizer.visualize(images, title_info, ncols=ncols, cell_size=cell_size)

            mock_viz_manager.individual_images.assert_called_once_with(
                images=images,
                title_info=title_info,
                ncols=ncols,
                cell_size=cell_size,
            )
