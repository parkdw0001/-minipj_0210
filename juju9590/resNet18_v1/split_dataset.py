import os, random, shutil
from pathlib import Path

random.seed(42)

# 🔥 Google Drive에 있는 실제 데이터 경로
DATA_ROOT = Path("/content/drive/MyDrive/data")

SRC_NORMAL = DATA_ROOT / "normal"
SRC_WRONG  = DATA_ROOT / "wrongway"

# ✅ 디버그 출력
print("NORMAL PATH:", SRC_NORMAL)
print("WRONG PATH :", SRC_WRONG)
print("NORMAL FILES:", list(SRC_NORMAL.glob("*"))[:5])
print("WRONG FILES :", list(SRC_WRONG.glob("*"))[:5])

OUT = Path(r"dataset")  # 최종 분리 폴더
splits = {"train":0.7, "val":0.15, "test":0.15}

def copy_split(src_dir, cls_name):
    files = [p for p in src_dir.glob("*.jpg")] + [p for p in src_dir.glob("*.png")] + [p for p in src_dir.glob("*.jpeg")]
    random.shuffle(files)

    n = len(files)
    n_train = int(n * splits["train"])
    n_val   = int(n * splits["val"])
    split_files = {
        "train": files[:n_train],
        "val":   files[n_train:n_train+n_val],
        "test":  files[n_train+n_val:]
    }

    for split, flist in split_files.items():
        dst = OUT / split / cls_name
        dst.mkdir(parents=True, exist_ok=True)
        for f in flist:
            shutil.copy2(f, dst / f.name)

    print(cls_name, {k:len(v) for k,v in split_files.items()})

def main():
    if OUT.exists():
        print("⚠️ dataset 폴더가 이미 있습니다. 지우고 다시 실행하거나 OUT 이름을 바꿔주세요.")
        return
    copy_split(SRC_NORMAL, "normal")
    copy_split(SRC_WRONG,  "wrongway")
    print("✅ done:", OUT.resolve())


if __name__ == "__main__":
    main()
