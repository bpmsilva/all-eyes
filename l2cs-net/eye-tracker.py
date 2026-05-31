import argparse
import pathlib
import tkinter as tk

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

from l2cs import select_device, Pipeline, render

CWD = pathlib.Path.cwd()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--cam", dest="cam_id", default=0, type=int)
    parser.add_argument("--arch", default="ResNet50", type=str)
    return parser.parse_args()

def get_yaw_pitch(results):
    yaw = results.yaw[0]
    pitch = results.pitch[0]

    if hasattr(yaw, "item"):
        yaw = yaw.item()
    if hasattr(pitch, "item"):
        pitch = pitch.item()

    return float(yaw), float(pitch)

class GazeApp:
    def __init__(self, root, gaze_pipeline, cap):
        self.root = root
        self.gaze_pipeline = gaze_pipeline
        self.cap = cap

        self.screen_w = root.winfo_screenwidth()
        self.screen_h = root.winfo_screenheight()

        self.canvas = tk.Canvas(
            root,
            width=self.screen_w,
            height=self.screen_h,
            bg="white"
        )
        self.canvas.pack()

        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", self.close)
        self.root.bind("<space>", self.save_calibration_point)

        ok, frame = self.cap.read()
        if not ok:
            raise IOError("Cannot read webcam frame")

        h, w, _ = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(
            "gravacao_l2cs_calibrado.mp4",
            fourcc,
            20.0,
            (w, h)
        )

        margin = 40

        self.calibration_points = [
            (margin, margin),
            (self.screen_w // 2, margin),
            (self.screen_w - margin, margin),

            (margin, self.screen_h // 2),
            (self.screen_w // 2, self.screen_h // 2),
            (self.screen_w - margin, self.screen_h // 2),

            (margin, self.screen_h - margin),
            (self.screen_w // 2, self.screen_h - margin),
            (self.screen_w - margin, self.screen_h - margin),
        ]

        self.current_point = 0
        self.calibrated = False

        self.X_train = []
        self.Y_train = []

        self.model_x = make_pipeline(
            PolynomialFeatures(degree=2),
            LinearRegression()
        )

        self.model_y = make_pipeline(
            PolynomialFeatures(degree=2),
            LinearRegression()
        )

        self.last_yaw = None
        self.last_pitch = None
        self.running = True

        print("Gravando em: gravacao_l2cs_calibrado.mp4")
        self.update()

    def draw_grid(self):
        self.canvas.create_line(
            self.screen_w // 2, 0,
            self.screen_w // 2, self.screen_h,
            fill="lightgray",
            width=2
        )

        self.canvas.create_line(
            0, self.screen_h // 2,
            self.screen_w, self.screen_h // 2,
            fill="lightgray",
            width=2
        )

    def draw_calibration(self):
        self.canvas.delete("all")
        self.draw_grid()

        x, y = self.calibration_points[self.current_point]

        self.canvas.create_oval(
            x - 20, y - 20,
            x + 20, y + 20,
            fill="red",
            outline="red"
        )

        self.canvas.create_text(
            self.screen_w // 2,
            50,
            text="Olhe para o ponto vermelho e aperte ESPAÇO",
            fill="black",
            font=("Arial", 24)
        )

        self.canvas.create_text(
            self.screen_w // 2,
            90,
            text=f"Ponto {self.current_point + 1}/{len(self.calibration_points)}",
            fill="black",
            font=("Arial", 18)
        )

    def draw_prediction(self, x, y):
        self.canvas.delete("all")
        self.draw_grid()

        self.canvas.create_oval(
            x - 18, y - 18,
            x + 18, y + 18,
            fill="red",
            outline="red"
        )

        self.canvas.create_text(
            self.screen_w // 2,
            50,
            text=f"Estimado: X={x} Y={y} | ESC para sair",
            fill="black",
            font=("Arial", 22)
        )

    def save_calibration_point(self, event=None):
        if self.calibrated:
            return

        if self.last_yaw is None or self.last_pitch is None:
            print("Ainda sem yaw/pitch válido.")
            return

        x, y = self.calibration_points[self.current_point]

        self.X_train.append([self.last_yaw, self.last_pitch])
        self.Y_train.append([x, y])

        print(
            f"Ponto {self.current_point + 1} salvo: "
            f"yaw={self.last_yaw:.4f}, pitch={self.last_pitch:.4f}, x={x}, y={y}"
        )

        self.current_point += 1

        if self.current_point >= len(self.calibration_points):
            X = np.array(self.X_train)
            Y = np.array(self.Y_train)

            self.model_x.fit(X, Y[:, 0])
            self.model_y.fit(X, Y[:, 1])

            self.calibrated = True
            print("Calibração concluída!")

    def update(self):
        if not self.running:
            return

        ok, frame = self.cap.read()

        if ok:
            with torch.no_grad():
                try:
                    results = self.gaze_pipeline.step(frame)
                except ValueError as e:
                    if "need at least one array to stack" in str(e):
                        print("Nenhum rosto detectado neste frame.")
                        self.root.after(10, self.update)
                        return
                    else:
                        raise e

                if results is None:
                    self.root.after(10, self.update)
                    return

                if len(results.yaw) == 0 or len(results.pitch) == 0:
                    self.root.after(10, self.update)
                    return

                yaw, pitch = get_yaw_pitch(results)

            self.last_yaw = yaw
            self.last_pitch = pitch

            camera_frame = render(frame, results)
            camera_frame = cv2.flip(camera_frame, 1)

            cv2.putText(
                camera_frame,
                f"Yaw={yaw:.3f} Pitch={pitch:.3f}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            self.out.write(camera_frame)

            cv2.imshow("Camera L2CS", camera_frame)

            if not self.calibrated:
                self.draw_calibration()
            else:
                pred_x = int(self.model_x.predict([[yaw, pitch]])[0])
                pred_y = int(self.model_y.predict([[yaw, pitch]])[0])

                pred_x = max(0, min(self.screen_w - 1, pred_x))
                pred_y = max(0, min(self.screen_h - 1, pred_y))

                self.draw_prediction(pred_x, pred_y)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.close()

        self.root.after(10, self.update)

    def close(self, event=None):
        self.running = False
        self.out.release()
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    args = parse_args()
    cudnn.enabled = True

    gaze_pipeline = Pipeline(
        weights=CWD / "models" / "L2CSNet_gaze360.pkl",
        arch=args.arch,
        device=select_device(args.device, batch_size=1)
    )

    cap = cv2.VideoCapture(args.cam_id)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    root = tk.Tk()
    app = GazeApp(root, gaze_pipeline, cap)
    root.mainloop()
