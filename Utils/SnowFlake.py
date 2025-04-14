# 在 module（例如 id_generator.py）中
import time
import threading

class Snowflake:
    # Snowflake 实现的代码不变
    def __init__(self, worker_id: int = 1, datacenter_id: int = 1, epoch: int = 1288834974657):
        self.worker_id_bits = 5
        self.datacenter_id_bits = 5
        self.sequence_bits = 12

        self.max_worker_id = -1 ^ (-1 << self.worker_id_bits)  # 最大 31
        self.max_datacenter_id = -1 ^ (-1 << self.datacenter_id_bits)  # 最大 31

        if not (0 <= worker_id <= self.max_worker_id):
            raise ValueError(f"worker_id 必须在 0 到 {self.max_worker_id} 之间")
        if not (0 <= datacenter_id <= self.max_datacenter_id):
            raise ValueError(f"datacenter_id 必须在 0 到 {self.max_datacenter_id} 之间")

        self.worker_id = worker_id
        self.datacenter_id = datacenter_id
        self.sequence = 0
        self.last_timestamp = -1
        self.epoch = epoch

        self.worker_id_shift = self.sequence_bits
        self.datacenter_id_shift = self.sequence_bits + self.worker_id_bits
        self.timestamp_left_shift = self.sequence_bits + self.worker_id_bits + self.datacenter_id_bits
        self.sequence_mask = -1 ^ (-1 << self.sequence_bits)

        self.lock = threading.Lock()

    def _current_time(self) -> int:
        return int(time.time() * 1000)

    def _wait_next_ms(self, last_timestamp: int) -> int:
        timestamp = self._current_time()
        while timestamp <= last_timestamp:
            timestamp = self._current_time()
        return timestamp

    def get_id(self) -> int:
        with self.lock:
            timestamp = self._current_time()
            if timestamp < self.last_timestamp:
                raise Exception("时钟回拨异常，无法生成 ID。")
            if timestamp == self.last_timestamp:
                self.sequence = (self.sequence + 1) & self.sequence_mask
                if self.sequence == 0:
                    timestamp = self._wait_next_ms(self.last_timestamp)
            else:
                self.sequence = 0
            self.last_timestamp = timestamp
            unique_id = ((timestamp - self.epoch) << self.timestamp_left_shift) | \
                        (self.datacenter_id << self.datacenter_id_shift) | \
                        (self.worker_id << self.worker_id_shift) | \
                        self.sequence
            return unique_id

# 全局实例，确保整个应用中都使用这个实例
snowflake_instance = Snowflake(worker_id=1, datacenter_id=1)
