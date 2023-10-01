import multiprocessing
import os
import sys

from scipy import signal

# now_dir = os.getcwd()
# sys.path.append(now_dir)
# print(sys.argv)
# inp_root = sys.argv[1]
# sr = int(sys.argv[2])
# n_p = int(sys.argv[3])
# exp_dir = sys.argv[4]
# noparallel = sys.argv[5] == "True"
# per = float(sys.argv[6])
import multiprocessing
import os
import traceback

import librosa
import numpy as np
from scipy.io import wavfile

from rvc.infer.lib.audio import load_audio
from rvc.infer.lib.slicer2 import Slicer

mutex = multiprocessing.Lock()

def println(strr, exp_dir):
    with open(f"{exp_dir}/preprocess.log", "a+", encoding="utf-8") as f:
        mutex.acquire()
        try:
            print(strr)
            f.write(f"{strr}\n")
            f.flush()
        finally:
            mutex.release()

class PreProcess:
    def __init__(self, sr, exp_dir, per=3.7):
        self.slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)
        self.per = per
        self.overlap = 0.3
        self.tail = self.per + self.overlap
        self.max = 0.9
        self.alpha = 0.75
        self.exp_dir = exp_dir
        self.gt_wavs_dir = f"{exp_dir}/0_gt_wavs"
        self.wavs16k_dir = f"{exp_dir}/1_16k_wavs"
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def norm_write(self, tmp_audio, idx0, idx1):
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.5:
            print(f"{idx0}-{idx1}-{tmp_max}-filtered")
            return
        tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (
            1 - self.alpha
        ) * tmp_audio
        wavfile.write(
            f"{self.gt_wavs_dir}/{idx0}_{idx1}.wav",
            self.sr,
            tmp_audio.astype(np.float32),
        )
        tmp_audio = librosa.resample(tmp_audio, orig_sr=self.sr, target_sr=16000)
        wavfile.write(
            f"{self.wavs16k_dir}/{idx0}_{idx1}.wav",
            16000,
            tmp_audio.astype(np.float32),
        )

    def pipeline(self, path, idx0, exp_dir):
        try:
            audio = load_audio(path, self.sr)
            # zero phased digital filter cause pre-ringing noise...
            # audio = signal.filtfilt(self.bh, self.ah, audio)
            audio = signal.lfilter(self.bh, self.ah, audio)

            idx1 = 0
            for audio in self.slicer.slice(audio):
                i = 0
                while 1:
                    start = int(self.sr * (self.per - self.overlap) * i)
                    i += 1
                    if len(audio[start:]) > self.tail * self.sr:
                        tmp_audio = audio[start : start + int(self.per * self.sr)]
                        self.norm_write(tmp_audio, idx0, idx1)
                        idx1 += 1
                    else:
                        tmp_audio = audio[start:]
                        idx1 += 1
                        break
                self.norm_write(tmp_audio, idx0, idx1)
            println(f"{path}->Suc.", exp_dir)
        except:
            println(f"{path}->{traceback.format_exc()}", exp_dir)

    def pipeline_mp(self, infos, exp_dir):
        for path, idx0 in infos:
            self.pipeline(path, idx0, exp_dir)

    def pipeline_mp_inp_dir(self, inp_root, noparallel, n_p, exp_dir):
        try:
            infos = [
                (f"{inp_root}/{name}", idx)
                for idx, name in enumerate(sorted(os.listdir(inp_root)))
            ]
            if noparallel:
                for i in range(n_p):
                    self.pipeline_mp(infos[i::n_p], exp_dir)
            else:
                ps = []
                for i in range(n_p):
                    p = multiprocessing.Process(
                        target=self.pipeline_mp, args=(infos[i::n_p], exp_dir)
                    )
                    ps.append(p)
                    p.start()
                for p in ps:
                    p.join()
        except:
            println(f"Fail. {traceback.format_exc()}", exp_dir)


def preprocess_trainset(inp_root, sr, n_p, exp_dir, noparallel, per):
    pp = PreProcess(sr, exp_dir, per)
    println("start preprocess", exp_dir)
    # println(sys.argv, exp_dir)
    pp.pipeline_mp_inp_dir(inp_root, noparallel, n_p, exp_dir)
    println("end preprocess", exp_dir)


if __name__ == "__main__":
    print("Don't run this as a standalone script")
