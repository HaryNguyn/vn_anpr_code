#!/usr/bin/env python3
"""Đánh giá nghiệm thu độ chính xác – chứng minh mục tiêu >= 95% có ý nghĩa thống kê.

Đầu vào:
  --gt   CSV: source,plate[,vehicle_type][,condition]   (plate ghi đúng như biển, hàm tự chuẩn hóa)
  --pred CSV do run_anpr.py sinh (ket_qua_anh.csv) hoặc CSV: source,candidate,conf,status
Chỉ số (theo 3 mức, xem README):
  - Độ chính xác toàn hệ thống (end-to-end): đúng / tổng biển thực tế (UNREAD và bỏ sót tính là sai)
  - Độ phủ tự động (coverage): tỷ lệ được ACCEPT
  - Độ chính xác trên tập chấp nhận (precision@accept)
  - CER (Levenshtein), khoảng tin cậy Wilson 95%, đường cong accuracy-coverage theo ngưỡng
Kết luận ĐẠT khi CẬN DƯỚI khoảng tin cậy 95% >= mục tiêu (mặc định 0.95).
"""
import argparse, csv, math
from collections import defaultdict


def norm(s):
    s = (s or "").upper().replace("Ð", "Đ")
    return "".join(c for c in s if c.isalnum())


def lev(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (ca != cb))
    return d[len(b)]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def load(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True); ap.add_argument("--pred", required=True)
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--key", default="source")
    a = ap.parse_args()
    gt = {r[a.key]: r for r in load(a.gt)}
    preds = defaultdict(list)
    for r in load(a.pred):
        preds[r[a.key]].append(r)

    rows = []
    for k, g in gt.items():
        truth = norm(g["plate"])
        cands = preds.get(k, [])
        # mỗi ảnh nghiệm thu chứa 1 biển mục tiêu: lấy dự đoán độ tin cậy cao nhất
        best = max(cands, key=lambda r: float(r.get("conf") or 0), default=None)
        text = norm(best.get("candidate") or best.get("text")) if best else ""
        conf = float(best.get("conf") or 0) if best else 0.0
        accepted = bool(best) and best.get("status", "ACCEPT") == "ACCEPT"
        rows.append(dict(truth=truth, text=text, conf=conf, accepted=accepted,
                         vtype=g.get("vehicle_type", "all"), cond=g.get("condition", "all")))

    def report(sub, title):
        n = len(sub)
        if n == 0:
            return
        e2e = sum(r["accepted"] and r["text"] == r["truth"] for r in sub)
        acc_n = sum(r["accepted"] for r in sub)
        raw = sum(r["text"] == r["truth"] for r in sub)
        cer = sum(lev(r["text"], r["truth"]) for r in sub) / max(1, sum(len(r["truth"]) for r in sub))
        lo, hi = wilson(e2e, n)
        plo, phi = wilson(e2e, acc_n) if acc_n else (0, 0)
        print(f"\n== {title}: n={n}")
        print(f"  Độ chính xác end-to-end (ACCEPT & đúng / tổng): {e2e/n:.4f}  CI95%=[{lo:.4f}, {hi:.4f}]"
              f"  -> {'ĐẠT' if lo >= a.target else 'CHƯA ĐẠT'} mục tiêu {a.target:.0%}")
        print(f"  Độ phủ tự động (ACCEPT/tổng): {acc_n/n:.4f}")
        if acc_n:
            print(f"  Độ chính xác trên tập ACCEPT: {e2e/acc_n:.4f}  CI95%=[{plo:.4f}, {phi:.4f}]")
        print(f"  Đọc đúng thô (bỏ qua ngưỡng): {raw/n:.4f}   CER: {cer:.4f}")

    report(rows, "TOÀN BỘ")
    for key in ("vtype", "cond"):
        vals = sorted({r[key] for r in rows})
        if len(vals) > 1:
            for v in vals:
                report([r for r in rows if r[key] == v], f"{key}={v}")

    print("\n== Đường cong accuracy-coverage (chỉ tính chuỗi khớp, theo ngưỡng conf)")
    print("  ngưỡng | độ phủ | chính xác trên phần phủ")
    for t in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98):
        cov = [r for r in rows if r["text"] and r["conf"] >= t]
        if cov:
            print(f"  {t:5.2f}  | {len(cov)/len(rows):.4f} | {sum(r['text']==r['truth'] for r in cov)/len(cov):.4f}")
    n_need = math.ceil((1.96 ** 2) * a.target * (1 - a.target) / (0.01 ** 2))
    print(f"\nGợi ý cỡ mẫu: để ước lượng tỷ lệ {a.target:.0%} với sai số ±1% cần ~{n_need} biển cho MỖI nhóm báo cáo.")


if __name__ == "__main__":
    main()
