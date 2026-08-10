import unittest

import torch

from lib.helpers.trainer_helper import CudaBatchPrefetcher


def _pinned_batch(batch_index):
    return (
        torch.full((2, 3), batch_index, dtype=torch.float32).pin_memory(),
        torch.full((2, 4), batch_index + 10, dtype=torch.float32).pin_memory(),
        {
            'depth': torch.full(
                (2, 5), batch_index + 20, dtype=torch.float32).pin_memory(),
            'nested': [torch.full(
                (2,), batch_index + 30, dtype=torch.int64).pin_memory()],
        },
        {'img_id': torch.tensor([batch_index, batch_index + 1])},
    )


class CudaBatchPrefetcherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA is required')
        cls.device = torch.device('cuda')

    def test_matches_blocking_transfer_across_two_epochs(self):
        copy_stream = torch.cuda.Stream(device=self.device)
        stream_id = copy_stream.cuda_stream

        for _ in range(2):
            host_batches = [_pinned_batch(index) for index in range(4)]
            expected = [
                (
                    batch[0].to(self.device),
                    batch[1].to(self.device),
                    {
                        'depth': batch[2]['depth'].to(self.device),
                        'nested': [batch[2]['nested'][0].to(self.device)],
                    },
                    batch[3],
                )
                for batch in host_batches
            ]
            prefetcher = CudaBatchPrefetcher(
                iter(host_batches), self.device, copy_stream=copy_stream)
            actual = list(prefetcher)
            prefetcher.close()

            self.assertEqual(copy_stream.cuda_stream, stream_id)
            self.assertEqual(len(actual), len(expected))
            for actual_batch, expected_batch in zip(actual, expected):
                torch.testing.assert_close(actual_batch[0], expected_batch[0])
                torch.testing.assert_close(actual_batch[1], expected_batch[1])
                torch.testing.assert_close(
                    actual_batch[2]['depth'], expected_batch[2]['depth'])
                torch.testing.assert_close(
                    actual_batch[2]['nested'][0],
                    expected_batch[2]['nested'][0])
                torch.testing.assert_close(
                    actual_batch[3]['img_id'], expected_batch[3]['img_id'])

    def test_rejects_unpinned_source_tensors(self):
        batch = (
            torch.zeros(1), torch.zeros(1), {'target': torch.zeros(1)}, {})
        with self.assertRaisesRegex(RuntimeError, 'to be pinned'):
            CudaBatchPrefetcher(
                iter([batch]), self.device,
                copy_stream=torch.cuda.Stream(device=self.device))


if __name__ == '__main__':
    unittest.main()
