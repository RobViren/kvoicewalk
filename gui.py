from __future__ import annotations
import os
import re


import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import scrolledtext, ttk


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OTHER_TEXT = (
    "If you mix vinegar, baking soda, and a bit of dish soap in a tall cylinder, "
    "the resulting eruption is both a visual and tactile delight, often used in "
    "classrooms to simulate volcanic activity on a miniature scale."
)

MODE_RANDOM_WALK = "Random Walk"
MODE_TEST_VOICE = "Test Voice"
MODE_TRANSCRIBE_MANY = "Transcribe Many"
MODE_EXPORT_BIN = "Export Voices Bin"
MODES = [MODE_RANDOM_WALK, MODE_TEST_VOICE, MODE_TRANSCRIBE_MANY, MODE_EXPORT_BIN]
PROGRESS_RE = re.compile(
    r"Progress:\s*step\s+(?P<step>\d+)/(?P<total>\d+)\s*\|.*?elapsed=(?P<elapsed>[0-9.]+)s(?:\s*\|\s*eta=(?P<eta>[0-9.]+)s)?"
)


@dataclass
class Task:
    task_id: int
    mode: str
    args: list[str]
    summary: str
    status: str = "Queued"
    step_limit: int | None = None


class KVoiceWalkGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("KVoiceWalk Queue")
        self.root.geometry("1280x860")

        self.tasks: list[Task] = []
        self.next_task_id = 1
        self.runner_thread: threading.Thread | None = None
        self.current_process: subprocess.Popen | None = None

        self.stop_requested = False
        self.events: queue.Queue[tuple] = queue.Queue()

        self.mode_var = tk.StringVar(value=MODE_RANDOM_WALK)
        self.device_var = tk.StringVar(value="cuda")
        self.target_audio_var = tk.StringVar(value="")
        self.voice_folder_var = tk.StringVar(value=str((PROJECT_ROOT / "voices").resolve()))
        self.starting_voice_var = tk.StringVar(value="")
        self.test_voice_var = tk.StringVar(value="")
        self.transcribe_many_var = tk.StringVar(value="")
        self.output_name_var = tk.StringVar(value="my_new_voice")
        self.population_limit_var = tk.StringVar(value="10")
        self.step_limit_var = tk.StringVar(value="10000")
        self.log_interval_var = tk.StringVar(value="10")
        self.interpolate_start_var = tk.BooleanVar(value=False)
        self.transcribe_start_var = tk.BooleanVar(value=False)

        self._build_ui()
        self._append_log(f"Project root: {PROJECT_ROOT}")
        self._append_log(f"Python: {sys.executable}")
        self.root.after(100, self._poll_events)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)
        self.root.rowconfigure(3, weight=1)

        config_frame = ttk.LabelFrame(self.root, text="Task Configuration", padding=10)
        config_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        for col in range(6):
            config_frame.columnconfigure(col, weight=1)

        ttk.Label(config_frame, text="Mode").grid(row=0, column=0, sticky="w")
        mode_combo = ttk.Combobox(config_frame, textvariable=self.mode_var, values=MODES, state="readonly")
        mode_combo.grid(row=0, column=1, sticky="ew", padx=(4, 8))

        ttk.Label(config_frame, text="Device").grid(row=0, column=2, sticky="w")
        device_combo = ttk.Combobox(
            config_frame,
            textvariable=self.device_var,
            values=["auto", "cpu", "cuda"],
            state="readonly",
        )
        device_combo.grid(row=0, column=3, sticky="ew", padx=(4, 8))

        ttk.Label(config_frame, text="Output Name").grid(row=0, column=4, sticky="w")
        ttk.Entry(config_frame, textvariable=self.output_name_var).grid(row=0, column=5, sticky="ew", padx=(4, 0))

        ttk.Label(config_frame, text="Target Audio").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(config_frame, textvariable=self.target_audio_var).grid(row=1, column=1, columnspan=4, sticky="ew", padx=(4, 8), pady=(8, 0))
        ttk.Button(config_frame, text="Browse", command=self._browse_target_audio).grid(row=1, column=5, sticky="ew", pady=(8, 0))

        ttk.Label(config_frame, text="Target Text").grid(row=2, column=0, sticky="nw", pady=(8, 0))
        self.target_text_widget = scrolledtext.ScrolledText(config_frame, height=4, wrap=tk.WORD)
        self.target_text_widget.grid(row=2, column=1, columnspan=5, sticky="ew", padx=(4, 0), pady=(8, 0))

        ttk.Label(config_frame, text="Other Text").grid(row=3, column=0, sticky="nw", pady=(8, 0))
        self.other_text_widget = scrolledtext.ScrolledText(config_frame, height=3, wrap=tk.WORD)
        self.other_text_widget.grid(row=3, column=1, columnspan=5, sticky="ew", padx=(4, 0), pady=(8, 0))
        self.other_text_widget.insert("1.0", DEFAULT_OTHER_TEXT)

        ttk.Label(config_frame, text="Voice Folder").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(config_frame, textvariable=self.voice_folder_var).grid(row=4, column=1, columnspan=4, sticky="ew", padx=(4, 8), pady=(8, 0))
        ttk.Button(config_frame, text="Browse", command=self._browse_voice_folder).grid(row=4, column=5, sticky="ew", pady=(8, 0))

        ttk.Label(config_frame, text="Starting Voice (.pt)").grid(row=5, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(config_frame, textvariable=self.starting_voice_var).grid(row=5, column=1, columnspan=4, sticky="ew", padx=(4, 8), pady=(8, 0))
        ttk.Button(config_frame, text="Browse", command=self._browse_starting_voice).grid(row=5, column=5, sticky="ew", pady=(8, 0))

        ttk.Label(config_frame, text="Test Voice (.pt)").grid(row=6, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(config_frame, textvariable=self.test_voice_var).grid(row=6, column=1, columnspan=4, sticky="ew", padx=(4, 8), pady=(8, 0))
        ttk.Button(config_frame, text="Browse", command=self._browse_test_voice).grid(row=6, column=5, sticky="ew", pady=(8, 0))

        ttk.Label(config_frame, text="Transcribe Many Path").grid(row=7, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(config_frame, textvariable=self.transcribe_many_var).grid(row=7, column=1, columnspan=4, sticky="ew", padx=(4, 8), pady=(8, 0))
        ttk.Button(config_frame, text="Browse", command=self._browse_transcribe_many).grid(row=7, column=5, sticky="ew", pady=(8, 0))

        ttk.Label(config_frame, text="Population Limit").grid(row=8, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(config_frame, from_=1, to=1000, textvariable=self.population_limit_var).grid(row=8, column=1, sticky="ew", padx=(4, 8), pady=(8, 0))

        ttk.Label(config_frame, text="Step Limit").grid(row=8, column=2, sticky="w", pady=(8, 0))
        ttk.Spinbox(config_frame, from_=1, to=10000000, textvariable=self.step_limit_var).grid(row=8, column=3, sticky="ew", padx=(4, 8), pady=(8, 0))

        ttk.Checkbutton(config_frame, text="Interpolate Start", variable=self.interpolate_start_var).grid(row=8, column=4, sticky="w", pady=(8, 0))
        ttk.Checkbutton(config_frame, text="Transcribe Start", variable=self.transcribe_start_var).grid(row=8, column=5, sticky="w", pady=(8, 0))

        ttk.Label(config_frame, text="Log Every N Steps").grid(row=9, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(config_frame, from_=1, to=100000, textvariable=self.log_interval_var).grid(row=9, column=1, sticky="ew", padx=(4, 8), pady=(8, 0))

        controls = ttk.Frame(config_frame)
        controls.grid(row=10, column=0, columnspan=6, sticky="ew", pady=(12, 0))
        controls.columnconfigure((0, 1, 2, 3, 4, 5), weight=1)

        ttk.Button(controls, text="Add Task", command=self._add_task).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(controls, text="Remove Selected", command=self._remove_selected).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(controls, text="Clear Queue", command=self._clear_queue).grid(row=0, column=2, sticky="ew", padx=6)
        tk.Button(
            controls,
            text="Start Queue",
            command=self._start_queue,
            bg="#1f8e3e",
            fg="white",
            activebackground="#166b2e",
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            relief=tk.RAISED,
            bd=2,
            padx=8,
            pady=4,
        ).grid(row=0, column=3, sticky="ew", padx=6)
        ttk.Button(controls, text="Stop Current", command=self._stop_queue).grid(row=0, column=4, sticky="ew", padx=6)
        ttk.Button(controls, text="Clear Log", command=self._clear_log).grid(row=0, column=5, sticky="ew", padx=(6, 0))

        queue_frame = ttk.LabelFrame(self.root, text="Queued Tasks", padding=10)
        queue_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
        queue_frame.columnconfigure(0, weight=1)
        queue_frame.rowconfigure(0, weight=1)

        self.task_tree = ttk.Treeview(
            queue_frame,
            columns=("id", "mode", "status", "eta", "summary"),
            show="headings",
            height=8,
        )
        self.task_tree.heading("id", text="ID")
        self.task_tree.heading("mode", text="Mode")
        self.task_tree.heading("status", text="Status")
        self.task_tree.heading("eta", text="ETA")
        self.task_tree.heading("summary", text="Summary")
        self.task_tree.column("id", width=60, anchor="center")
        self.task_tree.column("mode", width=140, anchor="center")
        self.task_tree.column("status", width=110, anchor="center")
        self.task_tree.column("eta", width=120, anchor="center")
        self.task_tree.column("summary", width=680, anchor="w")
        self.task_tree.grid(row=0, column=0, sticky="nsew")

        tree_scroll = ttk.Scrollbar(queue_frame, orient="vertical", command=self.task_tree.yview)
        self.task_tree.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.grid(row=0, column=1, sticky="ns")

        log_frame = ttk.LabelFrame(self.root, text="Execution Log", padding=10)
        log_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_widget = scrolledtext.ScrolledText(log_frame, height=12, wrap=tk.WORD)
        self.log_widget.grid(row=0, column=0, sticky="nsew")

    def _browse_target_audio(self) -> None:
        path = filedialog.askopenfilename(title="Select Target Audio", filetypes=[("Audio", "*.wav *.mp3 *.flac *.m4a"), ("All Files", "*.*")])
        if path:
            self.target_audio_var.set(path)

    def _browse_voice_folder(self) -> None:
        path = filedialog.askdirectory(title="Select Voice Folder")
        if path:
            self.voice_folder_var.set(path)

    def _browse_starting_voice(self) -> None:
        path = filedialog.askopenfilename(title="Select Starting Voice", filetypes=[("PyTorch", "*.pt"), ("All Files", "*.*")])
        if path:
            self.starting_voice_var.set(path)

    def _browse_test_voice(self) -> None:
        path = filedialog.askopenfilename(title="Select Test Voice", filetypes=[("PyTorch", "*.pt"), ("All Files", "*.*")])
        if path:
            self.test_voice_var.set(path)

    def _browse_transcribe_many(self) -> None:
        file_path = filedialog.askopenfilename(title="Select Audio File for --transcribe_many", filetypes=[("Audio", "*.wav *.mp3 *.flac *.m4a"), ("All Files", "*.*")])
        if file_path:
            self.transcribe_many_var.set(file_path)
            return

        dir_path = filedialog.askdirectory(title="Or Select Folder for --transcribe_many")
        if dir_path:
            self.transcribe_many_var.set(dir_path)

    def _read_text(self, widget: scrolledtext.ScrolledText) -> str:
        return widget.get("1.0", tk.END).strip()

    def _build_task(self) -> Task:
        mode = self.mode_var.get().strip()
        if mode not in MODES:
            raise ValueError("Select a valid mode")

        args: list[str] = []
        device = self.device_var.get().strip() or "auto"
        args += ["--device", device]

        step_limit: int | None = None

        output_name = self.output_name_var.get().strip()
        if output_name:
            args += ["--output_name", output_name]

        if mode == MODE_RANDOM_WALK:
            target_audio = self.target_audio_var.get().strip()
            target_text = self._read_text(self.target_text_widget)
            if not target_audio:
                raise ValueError("Random Walk requires Target Audio")
            if not Path(target_audio).exists():
                raise ValueError(f"Target Audio not found: {target_audio}")
            if not target_text and not self.transcribe_start_var.get():
                raise ValueError("Random Walk requires Target Text unless Transcribe Start is enabled")

            args += ["--target_audio", target_audio, "--target_text", target_text]

            other_text = self._read_text(self.other_text_widget)
            if other_text:
                args += ["--other_text", other_text]

            voice_folder = self.voice_folder_var.get().strip()
            if voice_folder:
                args += ["--voice_folder", voice_folder]

            starting_voice = self.starting_voice_var.get().strip()
            if starting_voice:
                args += ["--starting_voice", starting_voice]

            args += ["--population_limit", self.population_limit_var.get().strip() or "10"]
            step_limit = int(self.step_limit_var.get().strip() or "10000")
            if step_limit < 1:
                raise ValueError("Step Limit must be at least 1")
            log_interval = int(self.log_interval_var.get().strip() or "10")
            if log_interval < 1:
                raise ValueError("Log Every N Steps must be at least 1")
            args += ["--step_limit", str(step_limit)]
            args += ["--log_interval", str(log_interval)]

            if self.interpolate_start_var.get():
                args.append("--interpolate_start")
            if self.transcribe_start_var.get():
                args.append("--transcribe_start")

            summary = f"audio={Path(target_audio).name}, output={output_name or 'my_new_voice'}, device={device}, log_interval={log_interval}"

        elif mode == MODE_TEST_VOICE:
            test_voice = self.test_voice_var.get().strip()
            target_text = self._read_text(self.target_text_widget)
            if not test_voice:
                raise ValueError("Test Voice mode requires Test Voice (.pt)")
            if not Path(test_voice).exists():
                raise ValueError(f"Test Voice not found: {test_voice}")
            if not target_text:
                raise ValueError("Test Voice mode requires Target Text")

            args += ["--test_voice", test_voice, "--target_text", target_text]
            summary = f"test={Path(test_voice).name}, output={output_name or 'my_new_voice'}, device={device}"

        elif mode == MODE_TRANSCRIBE_MANY:
            transcribe_many = self.transcribe_many_var.get().strip()
            if not transcribe_many:
                raise ValueError("Transcribe Many mode requires a file or folder path")
            if not Path(transcribe_many).exists():
                raise ValueError(f"Transcribe Many path not found: {transcribe_many}")

            args += ["--transcribe_many", transcribe_many]
            summary = f"transcribe_many={Path(transcribe_many).name}, device={device}"

        elif mode == MODE_EXPORT_BIN:
            voice_folder = self.voice_folder_var.get().strip()
            if not voice_folder:
                raise ValueError("Export Voices Bin mode requires Voice Folder")
            if not Path(voice_folder).exists():
                raise ValueError(f"Voice Folder not found: {voice_folder}")

            args += ["--voice_folder", voice_folder, "--export_bin"]
            summary = f"export_bin from {Path(voice_folder).name}"

        else:
            raise ValueError("Unsupported mode")

        task = Task(task_id=self.next_task_id, mode=mode, args=args, summary=summary, step_limit=step_limit)
        self.next_task_id += 1
        return task

    def _add_task(self) -> None:
        try:
            task = self._build_task()
        except ValueError as e:
            messagebox.showerror("Invalid Task", str(e))
            return

        self.tasks.append(task)
        self.task_tree.insert("", tk.END, iid=str(task.task_id), values=(task.task_id, task.mode, task.status, "-", task.summary))
        self._append_log(f"Queued task #{task.task_id}: {task.mode} ({task.summary})")

    def _remove_selected(self) -> None:
        selected = self.task_tree.selection()
        if not selected:
            return

        selected_ids = {int(iid) for iid in selected}
        self.tasks = [task for task in self.tasks if task.task_id not in selected_ids or task.status == "Running"]

        for iid in selected:
            values = self.task_tree.item(iid, "values")
            if len(values) >= 3 and values[2] == "Running":
                continue
            self.task_tree.delete(iid)

    def _clear_queue(self) -> None:
        running_ids = {task.task_id for task in self.tasks if task.status == "Running"}
        self.tasks = [task for task in self.tasks if task.task_id in running_ids]

        for iid in self.task_tree.get_children():
            values = self.task_tree.item(iid, "values")
            if len(values) >= 3 and values[2] == "Running":
                continue
            self.task_tree.delete(iid)

    def _clear_log(self) -> None:
        self.log_widget.delete("1.0", tk.END)

    def _append_log(self, text: str) -> None:
        self.log_widget.insert(tk.END, text + "\n")
        self.log_widget.see(tk.END)

    def _set_task_status(self, task_id: int, status: str) -> None:
        for task in self.tasks:
            if task.task_id == task_id:
                task.status = status
                break

        iid = str(task_id)
        if self.task_tree.exists(iid):
            values = list(self.task_tree.item(iid, "values"))
            if len(values) >= 5:
                values[2] = status
                self.task_tree.item(iid, values=values)
        if status in {"Done", "Failed", "Stopped"}:
            self._set_task_eta(task_id, "-")

    def _set_task_eta(self, task_id: int, eta_text: str) -> None:
        iid = str(task_id)
        if self.task_tree.exists(iid):
            values = list(self.task_tree.item(iid, "values"))
            if len(values) >= 5:
                values[3] = eta_text
                self.task_tree.item(iid, values=values)

    @staticmethod
    def _format_eta(seconds: float) -> str:
        seconds = max(0, int(round(seconds)))
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours}h {minutes}m {secs}s"
        if minutes:
            return f"{minutes}m {secs}s"
        return f"{secs}s"

    def _start_queue(self) -> None:
        if self.runner_thread and self.runner_thread.is_alive():
            messagebox.showinfo("Queue Running", "Queue is already running")
            return

        has_queued = any(task.status == "Queued" for task in self.tasks)
        if not has_queued:
            messagebox.showinfo("No Tasks", "No queued tasks to run")
            return

        self.stop_requested = False
        self.runner_thread = threading.Thread(target=self._runner_loop, daemon=True)
        self.runner_thread.start()
        self._append_log("Queue started")

    def _stop_queue(self) -> None:
        self.stop_requested = True
        if self.current_process and self.current_process.poll() is None:
            try:
                self.current_process.terminate()
            except Exception as e:
                self._append_log(f"Failed to terminate current task: {e}")
        self._append_log("Stop requested")

    def _runner_loop(self) -> None:
        while True:
            if self.stop_requested:
                self.events.put(("queue_state", "stopped"))
                break

            next_task = next((task for task in self.tasks if task.status == "Queued"), None)
            if next_task is None:
                self.events.put(("queue_state", "completed"))
                break

            task = next_task
            self.events.put(("task_status", task.task_id, "Running"))
            self.events.put(("task_eta", task.task_id, "Estimating..."))
            cmd = [sys.executable, "main.py", *task.args]
            self.events.put(("log", f"\n=== Running task #{task.task_id}: {' '.join(cmd)}"))

            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            env = os.environ.copy()
            env.setdefault("PYTHONUTF8", "1")
            env.setdefault("PYTHONIOENCODING", "utf-8")
            return_code = -1

            try:
                self.current_process = subprocess.Popen(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=creationflags,
                    env=env,
                )

                assert self.current_process.stdout is not None
                for line in self.current_process.stdout:
                    line_text = line.rstrip("\n")
                    self.events.put(("log", line_text))
                    if task.mode == MODE_RANDOM_WALK:
                        match = PROGRESS_RE.search(line_text)
                        if match:
                            completed = int(match.group("step"))
                            total = int(match.group("total"))
                            elapsed = float(match.group("elapsed"))
                            eta_group = match.group("eta")
                            if eta_group is not None:
                                eta_seconds = float(eta_group)
                            elif completed > 0 and total >= completed:
                                eta_seconds = (elapsed / completed) * (total - completed)
                            else:
                                eta_seconds = 0.0
                            self.events.put(("task_eta", task.task_id, self._format_eta(eta_seconds)))
                    if self.stop_requested and self.current_process.poll() is None:
                        try:
                            self.current_process.terminate()
                        except Exception:
                            pass

                return_code = self.current_process.wait()

            except Exception as e:
                self.events.put(("log", f"Task #{task.task_id} failed to start: {e}"))
                return_code = -1

            finally:
                self.current_process = None

            if self.stop_requested:
                self.events.put(("task_status", task.task_id, "Stopped"))
                self.events.put(("task_eta", task.task_id, "-"))
                self.events.put(("log", f"Task #{task.task_id} stopped"))
                self.events.put(("queue_state", "stopped"))
                break

            if return_code == 0:
                self.events.put(("task_status", task.task_id, "Done"))
                self.events.put(("task_eta", task.task_id, "-"))
                self.events.put(("log", f"Task #{task.task_id} finished successfully"))
            else:
                self.events.put(("task_status", task.task_id, "Failed"))
                self.events.put(("task_eta", task.task_id, "-"))
                self.events.put(("log", f"Task #{task.task_id} failed with exit code {return_code}"))

    def _poll_events(self) -> None:
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break

            kind = event[0]
            if kind == "log":
                self._append_log(event[1])
            elif kind == "task_status":
                _, task_id, status = event
                self._set_task_status(task_id, status)
            elif kind == "task_eta":
                _, task_id, eta_text = event
                self._set_task_eta(task_id, eta_text)
            elif kind == "queue_state":
                state = event[1]
                if state == "completed":
                    self._append_log("Queue completed")
                elif state == "stopped":
                    self._append_log("Queue stopped")

        self.root.after(100, self._poll_events)


def main() -> None:
    root = tk.Tk()
    app = KVoiceWalkGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
