import torch


class AsyncPrefetcher:
    """Overlaps CPU→GPU data transfer with GPU computation.

    Uses a CUDA stream so that the next batch is copied to the GPU while
    the current batch is being processed by the model.
    """

    def __init__(self, data_iter, device: torch.device):
        self.data_iter = data_iter
        self.device = device
        self.stream = torch.cuda.Stream()
        self._next = None

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        if self._next is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            batch = self._next
        else:
            cpu_batch = next(self.data_iter)
            with torch.cuda.stream(self.stream):
                batch = {k: v.to(self.device, non_blocking=True) for k, v in cpu_batch.items()}
            torch.cuda.current_stream().wait_stream(self.stream)

        try:
            cpu_batch = next(self.data_iter)
            with torch.cuda.stream(self.stream):
                self._next = {k: v.to(self.device, non_blocking=True) for k, v in cpu_batch.items()}
        except StopIteration:
            self._next = None

        return batch
