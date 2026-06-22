import sys
import time
import struct
import threading
from collections import deque

import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QFrame, QSizePolicy, QGridLayout
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QColor, QPainterPath

from tof_detector import TOFDetector, unstuff_bytes, crc16, TOF
from gesture_detector import model as gesture_model, normalize_landmarks
import mediapipe as mp

COM_PORT    = "COM3"
STD_PRAG    = 100
CAM_TIMEOUT = 30

C_BG      = "#16161d"
C_SURFACE = "#1e1e28"
C_CARD    = "#23232f"
C_BORDER  = "#2e2e3e"
C_GREEN   = "#1D9E75"
C_AMBER   = "#EF9F27"
C_BLUE    = "#378ADD"
C_TEXT    = "#ececec"
C_MUTED   = "#666680"
C_DIM     = "#2a2a38"

GESTURE_EMOJI = {
    "alien":        ("👽", "Alien"),
    "korean_heart": ("🤟", "Korean heart"),
    "ok":           ("👌", "OK"),
    "peace":        ("✌️",  "Peace"),
    "surfer":       ("🤙", "Surfer"),
}

# ── NeuralNetwork ─────────────────────────────────────────────────────────────
class NeuralNetwork:
    def __init__(self):
        self.W1=self.b1=self.W2=self.b2=None
    def relu(self,x): return np.maximum(0,x)
    def softmax(self,x):
        x=x-np.max(x); e=np.exp(x); return e/np.sum(e)
    def predict(self,x):
        a1=self.relu(x@self.W1+self.b1)
        return np.argmax(self.softmax(a1@self.W2+self.b2))
    def load(self,path):
        d=np.load(path)
        self.W1=d["W1"];self.b1=d["b1"];self.W2=d["W2"];self.b2=d["b2"]
        return d["mean"],d["std"]

# ── ToF worker ────────────────────────────────────────────────────────────────
class TofSignals(QObject):
    new_distance = pyqtSignal(float)
    wave_detected = pyqtSignal()

class TofWorker(QThread):
    def __init__(self, port, prag):
        super().__init__()
        self.port=port; self.prag=prag
        self.signals=TofSignals()
        self.running=True
        self.window=deque(maxlen=20)
        self.last_valid=0
        self.wave_cooldown=0

    def run(self):
        import serial
        try:
            ser=serial.Serial(self.port,115200,timeout=0.1)
            ser.reset_input_buffer()
        except Exception as e:
            print(f"Serial: {e}"); return

        buf=bytearray()
        while self.running:
            data=ser.read(256)
            if data: buf.extend(data)
            if len(buf)>5000: buf.clear(); continue

            while True:
                idx=buf.find(b'\xFF\xFF')
                if idx==-1: break
                if len(buf)<idx+4: break
                unstuffed=unstuff_bytes(buf[idx+3:])
                if len(unstuffed)<6: break
                try: psz=struct.unpack('<H',unstuffed[4:6])[0]+1
                except: buf=buf[idx+2:]; break
                if psz>2000: buf=buf[idx+2:]; break
                tot=6+psz+2
                if len(unstuffed)<tot: break
                pay=unstuffed[:tot]
                if struct.unpack('<H',pay[-2:])[0]!=crc16(pay[:-2]):
                    buf=buf[idx+2:]; continue
                pos=6
                while pos<len(pay)-2:
                    cid=pay[pos]
                    csz=struct.unpack('<H',pay[pos+1:pos+3])[0]+1
                    cdata=pay[pos+4:pos+4+csz]
                    if cid==TOF and csz>=2:
                        d=struct.unpack('<H',cdata[0:2])[0]
                        if d==65535: d=self.last_valid
                        else: self.last_valid=d
                        self.window.append(d)
                        self.signals.new_distance.emit(float(d))
                        now=time.time()
                        if len(self.window)==20 and np.std(self.window)>self.prag and now>self.wave_cooldown:
                            self.wave_cooldown=now+CAM_TIMEOUT+2
                            self.signals.wave_detected.emit()
                    pos+=4+csz
                buf=buf[idx+2:]
        ser.close()

    def stop(self): self.running=False

# ── Camera worker ─────────────────────────────────────────────────────────────
class CamSignals(QObject):
    new_frame=pyqtSignal(np.ndarray,str,float)
    gesture_result=pyqtSignal(str,float)
    finished=pyqtSignal()

class CamWorker(QThread):
    def __init__(self,cap,timeout):
        super().__init__()
        self.cap=cap; self.timeout=timeout
        self.signals=CamSignals()
        self.running=True

    def run(self):
        mp_hands=mp.solutions.hands
        mp_draw=mp.solutions.drawing_utils
        mp_styles=mp.solutions.drawing_styles
        hands=mp_hands.Hands(static_image_mode=False,max_num_hands=1,
            min_detection_confidence=0.5,min_tracking_confidence=0.5)

        start=time.time(); best_label=None; best_conf=0.0

        while self.running and (time.time()-start)<self.timeout:
            ok,frame=self.cap.read()
            if not ok: continue
            frame=cv2.flip(frame,1)
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            res=hands.process(rgb)
            label=""; conf=0.0
            if res.multi_hand_landmarks:
                for hl in res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame,hl,mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style())
                    lm=np.array([[p.x,p.y,p.z] for p in hl.landmark])
                    lm,scale=normalize_landmarks(lm)
                    if scale<0.25: continue
                    lm/=scale
                    proba=gesture_model.predict_proba(lm.flatten().reshape(1,-1))[0]
                    i=np.argmax(proba); conf=float(proba[i]); label=gesture_model.classes_[i]
                    if conf>best_conf: best_conf=conf; best_label=label
            self.signals.new_frame.emit(frame,label,conf)

        self.signals.gesture_result.emit(best_label or "",best_conf)
        self.signals.finished.emit()

    def stop(self): self.running=False

# ── ToF graf ──────────────────────────────────────────────────────────────────
class TofGraph(QWidget):
    def __init__(self):
        super().__init__()
        self.data=deque(maxlen=120)
        self.setMinimumHeight(160)
        self.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)

    def add_point(self,v):
        self.data.append(v); self.update()

    def paintEvent(self,_):
        p=QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w,h=self.width(),self.height()
        p.fillRect(0,0,w,h,QColor(C_DIM))

        # grid lines
        p.setPen(QPen(QColor("#1e1e2e"),1,Qt.DashLine))
        for frac in [0.25,0.5,0.75]:
            yy=int(h*frac); p.drawLine(0,yy,w,yy)

        if len(self.data)<2: return

        mx=800.0; pts=list(self.data); n=len(pts)
        xs=[int(i/(n-1)*w) for i in range(n)]
        ys=[int(h-(v/mx)*h) for v in pts]

        path=QPainterPath()
        path.moveTo(xs[0],h)
        for x,y in zip(xs,ys): path.lineTo(x,y)
        path.lineTo(xs[-1],h); path.closeSubpath()
        fill=QColor(C_GREEN); fill.setAlpha(30)
        p.fillPath(path,fill)

        p.setPen(QPen(QColor(C_GREEN),2))
        for i in range(1,n): p.drawLine(xs[i-1],ys[i-1],xs[i],ys[i])

        # current value
        if pts:
            p.setPen(QColor(C_TEXT))
            p.setFont(QFont("Segoe UI",12,QFont.Bold))
            p.drawText(10,22,f"{int(pts[-1])} mm")

            # y labels
            p.setFont(QFont("Segoe UI",9))
            p.setPen(QColor(C_MUTED))
            p.drawText(10,h-6,"0")
            p.drawText(10,int(h*0.5)+4,"400")
            p.drawText(10,14,"800")
        p.end()

# ── Gesture panel (emoji + ime) ───────────────────────────────────────────────
class GesturePanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        lay=QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(6)

        self.emoji_lbl=QLabel("?")
        self.emoji_lbl.setAlignment(Qt.AlignCenter)
        self.emoji_lbl.setFont(QFont("Segoe UI Emoji",64))
        self.emoji_lbl.setStyleSheet(f"color:{C_MUTED}; border:none;")

        self.name_lbl=QLabel("Pokaži gesto")
        self.name_lbl.setAlignment(Qt.AlignCenter)
        self.name_lbl.setFont(QFont("Segoe UI",16,QFont.Bold))
        self.name_lbl.setStyleSheet(f"color:{C_MUTED}; border:none;")

        self.conf_lbl=QLabel("")
        self.conf_lbl.setAlignment(Qt.AlignCenter)
        self.conf_lbl.setFont(QFont("Segoe UI",11))
        self.conf_lbl.setStyleSheet(f"color:{C_MUTED}; border:none;")

        # prikaz vseh možnih gest
        grid_widget=QWidget()
        grid=QHBoxLayout(grid_widget)
        grid.setSpacing(10)
        grid.setAlignment(Qt.AlignCenter)
        for key,(emoji,name) in GESTURE_EMOJI.items():
            pill=QLabel(f"{emoji}  {name}")
            pill.setFont(QFont("Segoe UI Emoji",11))
            pill.setStyleSheet(f"""
                background:{C_DIM}; color:{C_MUTED};
                border-radius:16px; padding:5px 14px; border:none;
            """)
            grid.addWidget(pill)
        self.pills=grid

        lay.addStretch()
        lay.addWidget(self.emoji_lbl)
        lay.addWidget(self.name_lbl)
        lay.addWidget(self.conf_lbl)
        lay.addSpacing(16)
        lay.addWidget(grid_widget)
        lay.addStretch()

    def set_gesture(self,label,conf):
        if label and label in GESTURE_EMOJI:
            emoji,name=GESTURE_EMOJI[label]
            self.emoji_lbl.setText(emoji)
            self.emoji_lbl.setStyleSheet(f"color:{C_TEXT}; border:none;")
            self.name_lbl.setText(name)
            self.name_lbl.setStyleSheet(f"color:{C_TEXT}; border:none;")
            self.conf_lbl.setText(f"{int(conf*100)}% zaupanje")
            self.conf_lbl.setStyleSheet(f"color:{C_GREEN}; border:none;")
        else:
            self.emoji_lbl.setText("?")
            self.emoji_lbl.setStyleSheet(f"color:{C_MUTED}; border:none;")
            self.name_lbl.setText("Pokaži gesto")
            self.name_lbl.setStyleSheet(f"color:{C_MUTED}; border:none;")
            self.conf_lbl.setText("")

    def reset(self):
        self.set_gesture(None,0)

# ── Glavno okno ───────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesture Detector")
        self.resize(1100,660)
        self.setStyleSheet(f"background:{C_BG}; color:{C_TEXT}; font-family:'Segoe UI';")

        self.cap=cv2.VideoCapture(0)
        for _ in range(5): self.cap.read()

        self.cam_worker=None
        self.cam_active=False
        self.cam_end_time=None

        self._build_ui()
        self._start_tof()

        self.idle_timer=QTimer()
        self.idle_timer.timeout.connect(self._update_idle_cam)
        self.idle_timer.start(33)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _card(self,title=""):
        f=QFrame()
        f.setStyleSheet(f"""
            QFrame{{background:{C_CARD};border:0.5px solid {C_BORDER};border-radius:14px;}}
        """)
        lay=QVBoxLayout(f)
        lay.setContentsMargins(14,10,14,14)
        lay.setSpacing(8)
        if title:
            lbl=QLabel(title)
            lbl.setStyleSheet(f"color:{C_MUTED};font-size:11px;border:none;")
            lay.addWidget(lbl)
        return f,lay

    def _pill(self,text,bg,fg):
        l=QLabel(text)
        l.setStyleSheet(f"background:{bg};color:{fg};border-radius:12px;padding:3px 14px;font-size:12px;border:none;")
        return l

    def _metric_card(self,title):
        f=QFrame()
        f.setStyleSheet(f"background:{C_CARD};border:0.5px solid {C_BORDER};border-radius:12px;")
        lay=QVBoxLayout(f)
        lay.setContentsMargins(14,10,14,12)
        lay.setSpacing(2)
        t=QLabel(title)
        t.setStyleSheet(f"color:{C_MUTED};font-size:11px;border:none;")
        v=QLabel("–")
        v.setFont(QFont("Segoe UI",18,QFont.Bold))
        v.setStyleSheet(f"color:{C_TEXT};border:none;")
        s=QLabel("")
        s.setStyleSheet(f"color:{C_MUTED};font-size:11px;border:none;")
        lay.addWidget(t); lay.addWidget(v); lay.addWidget(s)
        return f,v,s

    # ── build ─────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root=QWidget(); self.setCentralWidget(root)
        main=QVBoxLayout(root)
        main.setContentsMargins(14,14,14,14); main.setSpacing(10)

        # header
        hdr=QFrame()
        hdr.setStyleSheet(f"background:{C_CARD};border-radius:12px;border:0.5px solid {C_BORDER};")
        hl=QHBoxLayout(hdr); hl.setContentsMargins(14,8,14,8)
        title=QLabel("Gesture Detector")
        title.setFont(QFont("Segoe UI",14,QFont.Bold))
        title.setStyleSheet("border:none;")
        self.status_pill=self._pill("čakam mahanje","#1a3a2e","#1D9E75")
        hl.addWidget(title); hl.addStretch(); hl.addWidget(self.status_pill)
        main.addWidget(hdr)

        # middle: kamera | gesture panel | tof
        mid=QHBoxLayout(); mid.setSpacing(10)

        # kamera
        cam_card,cam_lay=self._card("kamera")
        self.cam_label=QLabel()
        self.cam_label.setMinimumSize(360,270)
        self.cam_label.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setStyleSheet(f"background:#0e0e16;border-radius:8px;border:none;")
        cam_lay.addWidget(self.cam_label)
        mid.addWidget(cam_card,5)

        # gesture panel
        gest_card,gest_lay=self._card()
        self.gesture_panel=GesturePanel()
        gest_lay.addWidget(self.gesture_panel)
        mid.addWidget(gest_card,4)

        # tof
        tof_card,tof_lay=self._card("ToF signal – razdalja (mm)")
        self.tof_graph=TofGraph()
        tof_lay.addWidget(self.tof_graph)
        mid.addWidget(tof_card,3)

        main.addLayout(mid,1)

        # bottom metrics
        bot=QHBoxLayout(); bot.setSpacing(10)
        gf,self.gesture_val,self.gesture_sub=self._metric_card("zaznana gesta")
        sf,self.std_val,self.std_sub=self._metric_card("std signala")
        tf,self.state_val,self.state_sub=self._metric_card("stanje")
        bot.addWidget(gf); bot.addWidget(sf); bot.addWidget(tf)
        main.addLayout(bot)

        self.std_sub.setText(f"prag: {STD_PRAG} mm")
        self.state_val.setText("čakam mahanje")
        self.state_val.setStyleSheet(f"color:{C_AMBER};border:none;")

    # ── ToF ───────────────────────────────────────────────────────────────────
    def _start_tof(self):
        self.tof_worker=TofWorker(COM_PORT,STD_PRAG)
        self.tof_worker.signals.new_distance.connect(self._on_distance)
        self.tof_worker.signals.wave_detected.connect(self._on_wave)
        self.tof_worker.start()

    def _on_distance(self,d):
        self.tof_graph.add_point(d)
        if len(self.tof_worker.window)>1:
            std=float(np.std(self.tof_worker.window))
            self.std_val.setText(f"{int(std)} mm")

    def _on_wave(self):
        if self.cam_active: return
        self.cam_active=True
        self.cam_end_time=time.time()+CAM_TIMEOUT
        self.idle_timer.stop()
        self.gesture_panel.reset()

        self.status_pill.setText("kamera aktivna")
        self.status_pill.setStyleSheet("background:#0a2540;color:#378ADD;border-radius:12px;padding:3px 14px;font-size:12px;border:none;")
        self.state_val.setText("mahanje zaznano")
        self.state_val.setStyleSheet(f"color:{C_GREEN};font-size:18px;font-weight:500;border:none;")

        self.cam_worker=CamWorker(self.cap,CAM_TIMEOUT)
        self.cam_worker.signals.new_frame.connect(self._on_frame)
        self.cam_worker.signals.gesture_result.connect(self._on_gesture_result)
        self.cam_worker.signals.finished.connect(self._on_cam_finished)
        self.cam_worker.start()

    def _update_idle_cam(self):
        ok,frame=self.cap.read()
        if ok:
            frame=cv2.flip(frame,1)
            self._show_frame(frame)

    def _on_frame(self,frame,label,conf):
        self._show_frame(frame)
        if label and conf>0.5:
            self.gesture_panel.set_gesture(label,conf)
            self.gesture_val.setText(GESTURE_EMOJI[label][1] if label in GESTURE_EMOJI else label)
            self.gesture_sub.setText(f"{int(conf*100)}% zaupanje")
        if self.cam_end_time:
            rem=max(0,int(self.cam_end_time-time.time()))
            self.state_sub.setText(f"kamera odprta {rem}s")

    def _show_frame(self,frame):
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        h,w,ch=rgb.shape
        img=QImage(rgb.data,w,h,ch*w,QImage.Format_RGB888)
        pix=QPixmap.fromImage(img).scaled(
            self.cam_label.width(),self.cam_label.height(),
            Qt.KeepAspectRatio,Qt.SmoothTransformation)
        self.cam_label.setPixmap(pix)

    def _on_gesture_result(self,label,conf):
        if label:
            self.gesture_panel.set_gesture(label,conf)
            self.gesture_val.setText(GESTURE_EMOJI[label][1] if label in GESTURE_EMOJI else label)
            self.gesture_sub.setText(f"{int(conf*100)}% zaupanje")

    def _on_cam_finished(self):
        self.cam_active=False
        self.status_pill.setText("čakam mahanje")
        self.status_pill.setStyleSheet("background:#1a3a2e;color:#1D9E75;border-radius:12px;padding:3px 14px;font-size:12px;border:none;")
        self.state_val.setText("čakam mahanje")
        self.state_val.setStyleSheet(f"color:{C_AMBER};font-size:18px;font-weight:500;border:none;")
        self.state_sub.setText("")
        self.idle_timer.start(33)

    def closeEvent(self,e):
        self.tof_worker.stop(); self.tof_worker.wait()
        if self.cam_worker: self.cam_worker.stop(); self.cam_worker.wait()
        self.cap.release(); cv2.destroyAllWindows(); e.accept()

if __name__=="__main__":
    app=QApplication(sys.argv)
    app.setStyle("Fusion")
    win=MainWindow(); win.show()
    sys.exit(app.exec_())