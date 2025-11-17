import struct
import threading
import time
import random
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    import serial
    import serial.tools.list_ports
    UART_AVAILABLE = True
except ImportError:
    UART_AVAILABLE = False

FRAME_SIZE = 16 * 4 + 2
TERMINATOR = b'\xAA\xBB'

class UARTReader(threading.Thread):
    def __init__(self, callback, port=None, baudrate=115200, dummy=False):
        super().__init__(daemon=True)
        self.callback = callback
        self.dummy = dummy or not UART_AVAILABLE
        self.running = False
        self.port = port
        self.baudrate = baudrate
        self.ser = None

    def run(self):
        self.running = True
        if self.dummy:
            self._run_dummy()
        else:
            try:
                self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
                self._run_uart()
            except Exception as e:
                print(f"UART open error: {e}")
                self.running = False

    def _run_uart(self):
        buffer = b''
        while self.running:
            data = self.ser.read(FRAME_SIZE)
            if data:
                buffer += data
                while len(buffer) >= FRAME_SIZE:
                    frame, buffer = buffer[:FRAME_SIZE], buffer[FRAME_SIZE:]
                    if frame[-2:] == TERMINATOR:
                        floats = struct.unpack('<16f', frame[:64])
                        self.callback(floats)

    def _run_dummy(self):
        while self.running:
            dummy_floats = [random.uniform(0, 10) for _ in range(16)]
            self.callback(dummy_floats)
            time.sleep(0.1)

    def stop(self):
        self.running = False
        if self.ser:
            self.ser.close()

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("UART Data Visualizer")
        self.config_data = None
        self.lines = []
        self.subplots = []

        self.reader = None
        # id отложенного вызова update_plot (для отмены при закрытии)
        self._after_id = None
        # порядок ключей каналов для корректного маппинга
        self.channel_keys = []
        # флаг закрытия — предотвращает повторную регистрацию after при завершении
        self._closing = False

        # Menu
        menubar = tk.Menu(root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Config", command=self.load_config)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_close)
        menubar.add_cascade(label="File", menu=file_menu)
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", "UART Visualizer v1.1 with JSON Config"))
        menubar.add_cascade(label="Help", menu=help_menu)
        root.config(menu=menubar)

        # Controls frame
        ctrl_frame = ttk.Frame(root)
        ctrl_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(ctrl_frame, text="Port:").pack(side=tk.LEFT)
        self.port_var = tk.StringVar()
        self.port_box = ttk.Combobox(ctrl_frame, textvariable=self.port_var, width=10)
        self.port_box.pack(side=tk.LEFT, padx=5)
        self.refresh_ports()

        ttk.Label(ctrl_frame, text="Baud:").pack(side=tk.LEFT)
        self.baud_var = tk.StringVar(value="115200")
        self.baud_box = ttk.Entry(ctrl_frame, textvariable=self.baud_var, width=8)
        self.baud_box.pack(side=tk.LEFT, padx=5)

        self.dummy_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl_frame, text="Dummy Mode", variable=self.dummy_var).pack(side=tk.LEFT, padx=5)

        self.start_btn = ttk.Button(ctrl_frame, text="Start", command=self.start_reader)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(ctrl_frame, text="Stop", command=self.stop_reader, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Data storage
        self.history_length = 200
        self.data = np.zeros((16, self.history_length))

        # Placeholder figure
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.text(0.5, 0.5, "Load config to start", ha='center', va='center')
        canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas = canvas

        self.update_plot()

    def load_config(self):
        path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if not path:
            return
        with open(path, 'r') as f:
            self.config_data = json.load(f)
        self.setup_plots()

    def setup_plots(self):
        if not self.config_data:
            return

        subplots_cfg = self.config_data.get("subplots", [])
        channels_cfg = self.config_data.get("channels", {})

        self.fig.clf()
        self.subplots = []
        self.lines = []

        # гарантируем хотя бы 1 subplot
        n_subplots = len(subplots_cfg) or 1
        for i in range(n_subplots):
            subplot_cfg = subplots_cfg[i] if i < len(subplots_cfg) else {}
            ax = self.fig.add_subplot(n_subplots, 1, i+1)
            ax.set_title(subplot_cfg.get("title", f"Plot {i+1}"))
            ax.set_ylabel(subplot_cfg.get("y_label", ""))
            self.subplots.append(ax)

        # сохраним порядок ключей и корректно привяжем индекс канала
        self.channel_keys = list(channels_cfg.keys())
        for ch_idx, ch_name in enumerate(self.channel_keys):
            ch = channels_cfg[ch_name]
            subplot_idx = ch.get("subplot", 0)
            # защита от выхода за границы
            if subplot_idx < 0:
                subplot_idx = 0
            if subplot_idx >= len(self.subplots):
                subplot_idx = len(self.subplots) - 1
            ax = self.subplots[subplot_idx]
            line, = ax.plot(np.zeros(self.history_length), label=f"{ch.get('name', ch_name)} [{ch.get('unit','')}]")
            # сохраняем индекс канала для обновления данных
            self.lines.append((line, ch, ch_idx))
            ax.legend(fontsize=6, ncol=4)

        self.fig.tight_layout()
        self.canvas.draw()

    def refresh_ports(self):
        if UART_AVAILABLE:
            ports = [p.device for p in serial.tools.list_ports.comports()]
        else:
            ports = []
        self.port_box['values'] = ports
        if ports:
            self.port_var.set(ports[0])

    def start_reader(self):
        if self.reader and self.reader.running:
            return
        try:
            baud = int(self.baud_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid baud rate")
            return
        self.reader = UARTReader(self.on_new_data, port=self.port_var.get(), baudrate=baud, dummy=self.dummy_var.get())
        self.reader.start()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

    def stop_reader(self):
        if self.reader:
            self.reader.stop()
            try:
                # дождаться завершения потока (коротко)
                self.reader.join(timeout=1)
            except Exception:
                pass
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def on_new_data(self, floats):
        self.data = np.roll(self.data, -1, axis=1)
        self.data[:, -1] = floats

    def update_plot(self):
        # если идет процесс закрытия — не делаем ничего
        if getattr(self, "_closing", False):
            return
        if self.config_data and self.lines:
            for item in self.lines:
                # (line, cfg, ch_idx)
                line, cfg, ch_idx = item
                scaled = self.data[ch_idx] * cfg.get("scale", 1.0)
                line.set_ydata(scaled)
            for ax in self.subplots:
                ax.relim()
                ax.autoscale_view(scaley=True)
            try:
                self.canvas.draw()
            except tk.TclError:
                # виджет уже уничтожен
                return
        # безопасно планируем следующий вызов — сохраним id, чтобы отменить при закрытия
        try:
            self._after_id = self.root.after(100, self.update_plot)
        except tk.TclError:
            # root уже может быть уничтожен
            self._after_id = None

    def on_close(self):
        # выставляем флаг закрытия — блокируем дальнейшие after
        self._closing = True
        # отменим запланированный update_plot (если есть)
        if getattr(self, "_after_id", None):
            try:
                self.root.after_cancel(self._after_id)
            except Exception:
                pass
        # остановим reader и подождём его завершения
        if self.reader:
            try:
                self.reader.stop()
                self.reader.join(timeout=2)
            except Exception:
                pass
        # корректно завершаем mainloop и уничтожаем root
        try:
            self.root.quit()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
