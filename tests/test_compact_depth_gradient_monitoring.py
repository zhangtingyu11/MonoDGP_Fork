import unittest

from lib.helpers.swanlab_helper import chinese_grouped_monitoring
from lib.helpers.trainer_helper import depth_mean_clipping_payload


class CompactDepthGradientMonitoringTest(unittest.TestCase):
    def test_depth_mean_clipping_has_one_compact_dedicated_group(self):
        payload = depth_mean_clipping_payload({
            'depth_mean_clip_applied_fraction': 0.1,
            'depth_mean_pre_clip_max_absolute_gradient': 0.08,
            'depth_mean_clip_minimum_retained_fraction': 0.375,
            'depth_mean_clip_retained_energy_fraction': 0.92,
            'depth_mean_clip_prediction_count': 100,
        }, scope='训练中每5批')

        self.assertEqual(len(payload), 4)
        self.assertIn(
            '训练中每5批深度均值梯度裁剪/裁剪预测比例', payload)
        self.assertIn(
            '训练中每5批深度均值梯度裁剪/'
            '最严重裁剪预测的梯度保留比例', payload)

    def test_online_payload_keeps_layer_totals_without_layer_loss_fanout(self):
        raw = {
            'loss_ce': 1.0,
            'loss_depth': 2.0,
            'loss_ce_0': 3.0,
            'loss_depth_0': 4.0,
            'loss_ce_inter_0': 5.0,
            'monitor_depth_mae': 0.5,
            'monitor_depth_mae_0': 0.6,
            'monitor_depth_precision_gt_4_fraction': 0.25,
            'monitor_cardinality_gt_car_count': 2.0,
            'monitor_cardinality_gt_count': 2.0,
        }
        weights = {
            'loss_ce': 2.0,
            'loss_depth': 1.0,
            'loss_ce_0': 2.0,
            'loss_depth_0': 1.0,
            'loss_ce_inter_0': 2.0,
        }

        payload = chinese_grouped_monitoring(
            raw, weights, scope='训练中每5批',
            final_query_label='全部11组')

        self.assertIn(
            '训练中每5批辅助Decoder损失/辅助Decoder第1层/该层加权损失合计',
            payload)
        self.assertFalse(any(
            '辅助Decoder第1层/分类损失' in key for key in payload))
        self.assertIn(
            '训练中每5批深度诊断/最终Decoder层/'
            '深度置信精度大于4的匹配预测占比', payload)
        self.assertFalse(any(
            '辅助Decoder第1层/深度绝对误差' in key for key in payload))
        self.assertIn(
            '训练中每5批预测数量诊断/每张图真实车辆数', payload)
        self.assertFalse(any(
            key.endswith('/每张图真实目标数') for key in payload))


if __name__ == '__main__':
    unittest.main()
