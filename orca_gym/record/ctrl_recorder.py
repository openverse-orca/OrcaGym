import h5py

class CtrlRecorder:

    def __init__():
        

    class RecordState:
        RECORD = "record"
        REPLAY = "replay"
        REPLAY_FINISHED = "replay_finished"
        NONE = "none"

    def _load_record(self) -> None:
        if self.record_file is None:
            raise ValueError("record_file is not set.")
        
        # 读取record_file中的数据，存储到record_pool中
        with h5py.File(self.record_file, 'r') as f:
            if "float_data" in f:
                dset = f["float_data"]
                if self.record_cursor >= dset.shape[0]:
                    return False

                self.record_pool = dset[self.record_cursor:self.record_cursor + self.RECORD_POOL_SIZE].tolist()
                self.record_cursor += self.RECORD_POOL_SIZE
                return True

        return False
    
    def save_record(self) -> None:
        if self.record_state != CtrlRecorder.RecordState.RECORD:
            return
        
        if self.record_file is None:
            raise ValueError("record_file is not set.")

        with h5py.File(self.record_file, 'a') as f:
            # 如果数据集存在，获取其大小；否则，创建新的数据集
            if "float_data" in f:
                dset = f["float_data"]
                self.record_cursor = dset.shape[0]
            else:
                dset = f.create_dataset("float_data", (0, len(self.ctrl)), maxshape=(None, len(self.ctrl)), dtype='f', compression="gzip")
                self.record_cursor = 0

            # 将record_pool中的数据写入数据集
            dset.resize((self.record_cursor + len(self.record_pool), len(self.ctrl)))
            dset[self.record_cursor:] = np.array(self.record_pool)
            self.record_cursor += len(self.record_pool)
            self.record_pool.clear()

            print("Record saved.")


    def _replay(self) -> None:
        if self.record_state == CtrlRecorder.RecordState.REPLAY_FINISHED:
            return
        
        if len(self.record_pool) == 0:
            if not self._load_record():
                self.record_state = CtrlRecorder.RecordState.REPLAY_FINISHED
                print("Replay finished.")
                return

        self.ctrl = self.record_pool.pop(0)