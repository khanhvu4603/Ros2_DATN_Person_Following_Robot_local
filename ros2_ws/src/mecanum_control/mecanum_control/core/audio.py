import os
import threading
import subprocess
import time

class AudioManager:
    def __init__(self, logger=None):
        self.logger = logger
        self.lost_sound_thread = None
        self.stop_lost_sound_event = threading.Event()

    def _log_error(self, msg):
        if self.logger:
            self.logger.error(msg)
        else:
            print(f"[ERROR] {msg}")

    def _log_info(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(f"[INFO] {msg}")

    def _play_aplay(self, path):
        """Helper để play audio không blocking - giống y ảnh."""
        try:
            return subprocess.Popen(
                ["aplay", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            self._log_error("aplay not found")
            return None

    def play_sound_async(self, path, repeat=1):
        """Helper cho sound one-shot (enroll/run) - giống y ảnh."""
        if not os.path.exists(path):
            return
        def worker():
            for _ in range(repeat):
                proc = self._play_aplay(path)
                if proc is None:
                    return
                proc.wait()
        threading.Thread(target=worker, daemon=True).start()

    def _lost_sound_loop(self, sound_file):
        """Patched lost loop với terminate capability - giống y ảnh."""
        proc = None
        while not self.stop_lost_sound_event.is_set():
            if os.path.exists(sound_file):
                proc = self._play_aplay(sound_file)
                # chờ aplay xong hoặc stop
                while proc is not None and proc.poll() is None:
                    if self.stop_lost_sound_event.wait(0.1):
                        proc.terminate()
                        break
            # sleep có thể interrupt
            self.stop_lost_sound_event.wait(0.5)

    def start_lost_sound_loop(self, sound_file):
        if self.lost_sound_thread is not None and self.lost_sound_thread.is_alive():
            return
        
        self.stop_lost_sound_event.clear()
        self.lost_sound_thread = threading.Thread(target=self._lost_sound_loop, args=(sound_file,), daemon=True)
        self.lost_sound_thread.start()
        self._log_info("Started lost target sound loop.")

    def stop_lost_sound_loop(self):
        if self.lost_sound_thread is None or not self.lost_sound_thread.is_alive():
            return
        
        self.stop_lost_sound_event.set()
        self.lost_sound_thread.join(timeout=2.0)
        self.lost_sound_thread = None
        self._log_info("Stopped lost target sound loop.")
