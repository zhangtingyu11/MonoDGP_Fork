import unittest

import torch

from lib.helpers.tester_helper import CudaEvalBatchPrefetcher


def _pinned_batch(batch_index):
    return (
        torch.full((2, 3), batch_index, dtype=torch.float32).pin_memory(),
        torch.full((2, 4), batch_index + 10, dtype=torch.float32).pin_memory(),
        {'depth': torch.full((2, 5), batch_index + 20)},
        {
            'img_id': torch.tensor([batch_index, batch_index + 1]),
            'img_size': torch.full(
                (2, 2), batch_index + 30, dtype=torch.float32).pin_memory(),
        },
    )


class CudaEvalBatchPrefetcherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA is required')
        cls.device = torch.device('cuda')

    def test_matches_blocking_transfer_and_reuses_stream(self):
        copy_stream = torch.cuda.Stream(device=self.device)
        stream_id = copy_stream.cuda_stream

        for _ in range(2):
            host_batches = [_pinned_batch(index) for index in range(4)]
            prefetcher = CudaEvalBatchPrefetcher(
                iter(host_batches), self.device, copy_stream=copy_stream)
            actual = list(prefetcher)
            prefetcher.close()

            self.assertEqual(copy_stream.cuda_stream, stream_id)
            self.assertEqual(len(actual), len(host_batches))
            for gpu_batch, host_batch in zip(actual, host_batches):
                torch.testing.assert_close(
                    gpu_batch[0], host_batch[0].to(self.device))
                torch.testing.assert_close(
                    gpu_batch[1], host_batch[1].to(self.device))
                torch.testing.assert_close(
                    gpu_batch[4], host_batch[3]['img_size'].to(self.device))
                self.assertIs(gpu_batch[2], host_batch[2])
                self.assertIs(gpu_batch[3], host_batch[3])

    def test_rejects_unpinned_source_tensors(self):
        batch = (
            torch.zeros(1), torch.zeros(1), {},
            {'img_size': torch.zeros(1)})
        with self.assertRaisesRegex(RuntimeError, 'pinned source tensors'):
            CudaEvalBatchPrefetcher(
                iter([batch]), self.device,
                copy_stream=torch.cuda.Stream(device=self.device))


if __name__ == '__main__':
    unittest.main()
