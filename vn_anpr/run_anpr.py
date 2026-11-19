#!/usr/bin/env python3
"""Chạy nhận diện biển số.
Ví dụ:
  python run_anpr.py --config config.yaml --images data/anh/ --out ketqua/            # nhiều ảnh
  python run_anpr.py --config config.yaml --images data/xe_theo_thu_muc/ --group-by-folder --out ketqua/
  python run_anpr.py --config config.yaml --video data/cong_vao.mp4 --save-video --out ketqua/
"""
import argparse, csv, json, os, time
from anpr.pipeline import ANPRPipeline, load_config, list_images


def write_csv(rows, path):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser(description="VN-ANPR – nhận diện biển số ô tô, xe máy Việt Nam")
    ap.add_argument("--config", default="config.yaml")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--images", help="ảnh hoặc thư mục ảnh")
    g.add_argument("--video", help="file video hoặc URL RTSP")
    ap.add_argument("--group-by-folder", action="store_true", help="bỏ phiếu các ảnh cùng thư mục (cùng 1 xe)")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--out", default="ketqua")
    ap.add_argument("--save-crops", default=None, help="thư mục lưu crop biển để gán nhãn, huấn luyện lại")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pipe = ANPRPipeline(load_config(a.config))
    pipe.crop_dir = a.save_crops
    t0 = time.time()
    if a.images:
        res = pipe.process_images(list_images(a.images), a.group_by_folder)
        write_csv(res["images"], os.path.join(a.out, "ket_qua_anh.csv"))
        if "groups" in res:
            write_csv(res["groups"], os.path.join(a.out, "ket_qua_nhom.csv"))
        n = len(res["images"])
    else:
        outv = os.path.join(a.out, "video_gan_nhan.mp4") if a.save_video else None
        res = pipe.process_video(a.video, outv)
        write_csv(res["vehicles"], os.path.join(a.out, "ket_qua_video.csv"))
        n = res["frames"]
    with open(os.path.join(a.out, "ket_qua.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    dt = time.time() - t0
    print(f"Hoàn thành {n} mục trong {dt:.1f}s ({n / max(dt, 1e-6):.1f} mục/s). Kết quả: {a.out}/")


if __name__ == "__main__":
    main()
