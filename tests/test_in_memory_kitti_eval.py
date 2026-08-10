import numpy as np
import sys
import types


def _prediction_from_anno(anno, class_to_id, index):
    dimensions_lhw = anno['dimensions'][index]
    dimensions_hwl = dimensions_lhw[[1, 2, 0]]
    return [
        class_to_id[anno['name'][index]],
        anno['alpha'][index],
        *anno['bbox'][index],
        *dimensions_hwl,
        *anno['location'][index],
        anno['rotation_y'][index],
        anno['score'][index],
    ]


def test_in_memory_annotations_match_historical_text_roundtrip(tmp_path):
    # This contract tests only the annotation conversion.  Avoid importing the
    # historical evaluator, which initializes CUDA as an import side effect.
    eval_name = 'lib.datasets.kitti.kitti_eval_python.eval'
    eval_stub = types.ModuleType(eval_name)
    eval_stub.get_official_eval_result = lambda *_args, **_kwargs: None
    eval_stub.get_distance_eval_result = lambda *_args, **_kwargs: None
    sys.modules[eval_name] = eval_stub
    from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
    from lib.datasets.kitti.kitti_eval_python import kitti_common

    text_dir = tmp_path / 'predictions'
    text_dir.mkdir()
    (text_dir / '000001.txt').write_text(
        'Car 0.0 0 -0.12 10.13 20.12 30.12 40.13 '
        '1.50 1.60 3.90 4.00 5.00 20.01 0.78 0.88\n',
        encoding='utf-8',
    )
    (text_dir / '000002.txt').write_text('', encoding='utf-8')
    disk_annos = kitti_common.get_label_annos(text_dir, [1, 2])

    dataset = object.__new__(KITTI_Dataset)
    dataset.idx_list = ['000001', '000002']
    dataset.class_name = ['Pedestrian', 'Car', 'Cyclist']
    class_to_id = {name: index for index, name in enumerate(dataset.class_name)}
    decoded = {
        1: [_prediction_from_anno(disk_annos[0], class_to_id, 0)],
        2: [],
    }
    memory_annos = dataset._decoded_predictions_to_annos(decoded)

    for disk, memory in zip(disk_annos, memory_annos):
        assert disk.keys() == memory.keys()
        for key in disk:
            np.testing.assert_array_equal(memory[key], disk[key])
