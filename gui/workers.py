"""
gui/workers.py - 后台任务线程

约定：核心层耗时函数接受可选参数
    progress_cb(percent:int, message:str)   进度回调
训练任务额外支持
    stop_event: threading.Event             置位后当前 epoch 结束即停止
所有回调都经 Qt 信号转发，GUI 更新发生在主线程（线程安全）。
"""

import threading
import traceback

from PySide6.QtCore import QThread, Signal


class TaskWorker(QThread):
    """通用后台任务。fn 需接受 progress_cb 关键字参数（可忽略不用）。"""

    progress = Signal(int, str)      # (percent, message)
    succeeded = Signal(object)       # 任务返回值
    failed = Signal(str)             # 错误信息

    def __init__(self, fn, *args, parent=None, **kwargs):
        super().__init__(parent)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.stop_event = threading.Event()
        self.result = None

    def _progress_cb(self, percent, message=''):
        self.progress.emit(int(percent), str(message))

    def run(self):
        try:
            import inspect
            kwargs = dict(self.kwargs)
            # 只传入目标函数声明了的回调参数，避免 TypeError
            params = inspect.signature(self.fn).parameters
            accepts_all = any(p.kind == p.VAR_KEYWORD for p in params.values())
            if accepts_all or 'progress_cb' in params:
                kwargs.setdefault('progress_cb', self._progress_cb)
            if accepts_all or 'stop_event' in params:
                kwargs.setdefault('stop_event', self.stop_event)
            self.result = self.fn(*self.args, **kwargs)
            self.succeeded.emit(self.result)
        except Exception:
            self.failed.emit(traceback.format_exc())

    def stop(self):
        self.stop_event.set()


class TrainWorker(QThread):
    """训练专用线程：额外携带逐 epoch 曲线信号。"""

    progress = Signal(int, str)
    epoch_end = Signal(dict, dict)   # (metrics, history)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, trainer, train_loader, val_loader, epochs,
                 save_kwargs=None, parent=None):
        super().__init__(parent)
        self.trainer = trainer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.save_kwargs = save_kwargs or {}
        self.stop_event = threading.Event()
        self.result = None

    def run(self):
        try:
            def on_epoch_end(epoch, metrics, history):
                pct = int((epoch + 1) / self.epochs * 100)
                self.progress.emit(pct, f'Epoch {epoch + 1}/{self.epochs}')
                self.epoch_end.emit(metrics, history)

            history, best, stopped = self.trainer.fit(
                self.train_loader, self.val_loader, self.epochs,
                on_epoch_end=on_epoch_end,
                should_stop=self.stop_event.is_set,
                **self.save_kwargs)
            self.result = {'history': history, 'best': best, 'stopped': stopped}
            self.succeeded.emit(self.result)
        except Exception:
            self.failed.emit(traceback.format_exc())

    def stop(self):
        self.stop_event.set()
